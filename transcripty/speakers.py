"""Speaker enrollment and identification via voice embeddings.

Supports three layers of speaker features:
1. Pyannote embedding (256-dim) — standard voice timbre
2. ECAPA-TDNN embedding (192-dim) — more discriminative for similar voices
3. Prosodic features (12-dim) — shimmer, jitter, pitch patterns

Combined scoring: weighted sum for better twin/sibling discrimination.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from transcripty.models import DiarizationResult

if TYPE_CHECKING:
    from transcripty.prosody import ProsodicFeatures

logger = logging.getLogger(__name__)

# Use numpy if available, pure Python fallback
try:
    import numpy as np

    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity using numpy."""
        va = np.array(a)
        vb = np.array(b)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

except ImportError:

    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity (pure Python fallback)."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class SpeakerProfile(BaseModel):
    """A stored voice profile for a known speaker."""

    embedding: list[float]
    ecapa_embedding: list[float] | None = None
    prosodic_features: list[float] | None = None
    enrolled_at: str


class SpeakerDB:
    """Database of known speaker voice profiles (embeddings).

    Supports enrolling speakers from reference audio and identifying
    speakers in diarization results by comparing embeddings.

    Usage:
        db = SpeakerDB()
        db.enroll("Alice", "alice_reference.mp3")
        db.save("speakers.json")

        db = SpeakerDB.load("speakers.json")
        names = db.identify(diarization_result)
        # {"SPEAKER_00": "Alice"}
    """

    def __init__(self) -> None:
        self.profiles: dict[str, SpeakerProfile] = {}

    def enroll(
        self,
        name: str,
        audio_path: str | Path,
        hf_token: str | None = None,
    ) -> None:
        """Enroll a speaker by extracting their voice embedding from reference audio.

        The reference audio should contain only the target speaker's voice.

        Args:
            name: Display name for this speaker.
            audio_path: Path to reference audio (any format).
            hf_token: HuggingFace token (falls back to HF_TOKEN env var).
        """
        from transcripty.diarize import diarize

        logger.info("Enrolling speaker '%s' from %s...", name, Path(audio_path).name)

        result = diarize(
            audio_path=audio_path,
            hf_token=hf_token,
            num_speakers=1,
        )

        if not result.embeddings:
            raise RuntimeError(
                f"No embeddings returned for '{name}'. "
                "Ensure pyannote pipeline supports embedding extraction."
            )

        # Take the first (and only) speaker embedding
        embedding = list(result.embeddings.values())[0]
        self.enroll_from_embedding(name, embedding)

    def enroll_from_embedding(
        self,
        name: str,
        embedding: list[float],
        ecapa_embedding: list[float] | None = None,
        prosodic_features: list[float] | None = None,
    ) -> None:
        """Enroll a speaker directly from a pre-computed embedding.

        Args:
            name: Display name for this speaker.
            embedding: Speaker embedding vector (pyannote, 256-dim).
            ecapa_embedding: Optional ECAPA-TDNN embedding (192-dim).
            prosodic_features: Optional prosodic feature vector (12-dim).
        """
        self.profiles[name] = SpeakerProfile(
            embedding=embedding,
            ecapa_embedding=ecapa_embedding,
            prosodic_features=prosodic_features,
            enrolled_at=datetime.now(timezone.utc).isoformat(),
        )
        layers = ["pyannote"]
        if ecapa_embedding:
            layers.append("ecapa")
        if prosodic_features:
            layers.append("prosody")
        logger.info("Enrolled '%s' with layers: %s (dim=%d)", name, "+".join(layers), len(embedding))

    def enroll_enhanced(
        self,
        name: str,
        audio_path: str | Path,
        hf_token: str | None = None,
    ) -> None:
        """Enroll a speaker with all three feature layers.

        Extracts pyannote embedding, ECAPA-TDNN embedding, and prosodic
        features from a single audio file.

        Args:
            name: Display name for this speaker.
            audio_path: Path to reference audio (should contain only this speaker).
            hf_token: HuggingFace token for pyannote.
        """
        from transcripty.diarize import diarize
        from transcripty.ecapa import extract_ecapa_embedding
        from transcripty.prosody import extract_prosodic_features

        audio_path = Path(audio_path)
        logger.info("Enhanced enrollment for '%s' from %s...", name, audio_path.name)

        # Layer 1: Pyannote embedding
        result = diarize(audio_path=audio_path, hf_token=hf_token, num_speakers=1)
        if not result.embeddings:
            raise RuntimeError(f"No pyannote embeddings for '{name}'.")
        pyannote_emb = list(result.embeddings.values())[0]

        # Layer 2: ECAPA-TDNN embedding
        ecapa_emb = extract_ecapa_embedding(audio_path)

        # Layer 3: Prosodic features
        prosodic = extract_prosodic_features(audio_path)

        self.enroll_from_embedding(
            name=name,
            embedding=pyannote_emb,
            ecapa_embedding=ecapa_emb,
            prosodic_features=prosodic.as_vector(),
        )

    def enroll_from_multiple_samples(
        self,
        name: str,
        audio_paths: list[str | Path],
        hf_token: str | None = None,
    ) -> None:
        """Enroll a speaker by averaging features across multiple audio samples.

        More robust than single-sample enrollment, especially for
        speakers with similar voices (twins, siblings).

        Args:
            name: Display name for this speaker.
            audio_paths: List of paths to reference audio files.
            hf_token: HuggingFace token for pyannote.
        """
        from transcripty.diarize import diarize
        from transcripty.ecapa import extract_ecapa_embedding
        from transcripty.prosody import extract_prosodic_features

        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for multi-sample enrollment")

        logger.info("Multi-sample enrollment for '%s' from %d files...", name, len(audio_paths))

        pyannote_embeddings = []
        ecapa_embeddings = []
        prosodic_vectors = []

        for audio_path in audio_paths:
            audio_path = Path(audio_path)
            if not audio_path.is_file():
                logger.warning("Skipping missing file: %s", audio_path)
                continue

            logger.info("  Processing %s...", audio_path.name)

            # Pyannote
            result = diarize(audio_path=audio_path, hf_token=hf_token, num_speakers=1)
            if result.embeddings:
                pyannote_embeddings.append(list(result.embeddings.values())[0])

            # ECAPA
            ecapa_emb = extract_ecapa_embedding(audio_path)
            ecapa_embeddings.append(ecapa_emb)

            # Prosody
            prosodic = extract_prosodic_features(audio_path)
            prosodic_vectors.append(prosodic.as_vector())

        if not pyannote_embeddings:
            raise RuntimeError(f"No pyannote embeddings extracted for '{name}'.")

        # Average and L2-normalize
        avg_pyannote = np.mean(pyannote_embeddings, axis=0)
        avg_pyannote = (avg_pyannote / np.linalg.norm(avg_pyannote)).tolist()

        avg_ecapa = None
        if ecapa_embeddings:
            avg_ecapa = np.mean(ecapa_embeddings, axis=0)
            avg_ecapa = (avg_ecapa / np.linalg.norm(avg_ecapa)).tolist()

        avg_prosody = None
        if prosodic_vectors:
            avg_prosody = np.mean(prosodic_vectors, axis=0).tolist()

        self.enroll_from_embedding(
            name=name,
            embedding=avg_pyannote,
            ecapa_embedding=avg_ecapa,
            prosodic_features=avg_prosody,
        )
        logger.info(
            "Enrolled '%s' from %d samples (pyannote=%d, ecapa=%d, prosody=%d)",
            name, len(audio_paths),
            len(pyannote_embeddings), len(ecapa_embeddings), len(prosodic_vectors),
        )

    def identify(
        self,
        result: DiarizationResult,
        threshold: float = 0.5,
        ecapa_embeddings: dict[str, list[float]] | None = None,
        prosodic_features: dict[str, list[float]] | None = None,
        weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
    ) -> dict[str, str]:
        """Identify speakers using exclusive greedy matching.

        Each speaker label is matched to at most one profile, and each
        profile is matched to at most one speaker label. Highest scoring
        pairs are matched first.

        Supports three scoring layers when available:
        1. Pyannote embedding similarity (always used)
        2. ECAPA-TDNN embedding similarity (if ecapa_embeddings provided
           AND profile has ecapa_embedding)
        3. Prosodic feature similarity (if prosodic_features provided
           AND profile has prosodic_features)

        Falls back to pyannote-only when enhanced features are unavailable.

        Args:
            result: DiarizationResult with embeddings from diarize().
            threshold: Minimum combined score to consider a match (0-1).
            ecapa_embeddings: Optional dict of speaker_label → ECAPA embedding.
            prosodic_features: Optional dict of speaker_label → prosodic vector.
            weights: (pyannote_weight, ecapa_weight, prosody_weight).
                Only active layers are weighted; weights are re-normalized.
                Default (0.4, 0.4, 0.2) when all three layers are available.

        Returns:
            Mapping of speaker labels to identified names.
        """
        if not result.embeddings:
            logger.warning("No embeddings in diarization result. Cannot identify speakers.")
            return {}

        if not self.profiles:
            logger.warning("No enrolled speakers. Cannot identify.")
            return {}

        w_pyannote, w_ecapa, w_prosody = weights

        # Compute all pairwise scores
        scores: list[tuple[float, str, str, dict]] = []
        for speaker_label, speaker_emb in result.embeddings.items():
            for name, profile in self.profiles.items():
                # Layer 1: Pyannote similarity (always available)
                pyannote_score = _cosine_similarity(speaker_emb, profile.embedding)

                # Determine active layers and compute weighted score
                active_weights = [w_pyannote]
                layer_scores = [pyannote_score]
                score_detail = {"pyannote": round(pyannote_score, 4)}

                # Layer 2: ECAPA similarity
                ecapa_score = None
                if (
                    ecapa_embeddings
                    and speaker_label in ecapa_embeddings
                    and profile.ecapa_embedding
                ):
                    ecapa_score = _cosine_similarity(
                        ecapa_embeddings[speaker_label], profile.ecapa_embedding,
                    )
                    active_weights.append(w_ecapa)
                    layer_scores.append(ecapa_score)
                    score_detail["ecapa"] = round(ecapa_score, 4)

                # Layer 3: Prosodic similarity
                prosody_score = None
                if (
                    prosodic_features
                    and speaker_label in prosodic_features
                    and profile.prosodic_features
                ):
                    prosody_score = _cosine_similarity(
                        prosodic_features[speaker_label], profile.prosodic_features,
                    )
                    active_weights.append(w_prosody)
                    layer_scores.append(prosody_score)
                    score_detail["prosody"] = round(prosody_score, 4)

                # Weighted average (re-normalize weights to sum to 1)
                total_weight = sum(active_weights)
                if total_weight > 0:
                    combined = sum(
                        w * s for w, s in zip(active_weights, layer_scores)
                    ) / total_weight
                else:
                    combined = pyannote_score

                score_detail["combined"] = round(combined, 4)
                score_detail["layers"] = len(active_weights)
                scores.append((combined, speaker_label, name, score_detail))

        # Greedy exclusive matching: highest combined score first
        # When ECAPA is available and two candidates are close (<0.05 difference),
        # use ECAPA as tiebreaker instead of pyannote
        scores.sort(reverse=True, key=lambda x: x[0])
        matched_speakers: set[str] = set()
        matched_profiles: set[str] = set()
        matches: dict[str, str] = {}

        for combined, speaker_label, name, detail in scores:
            if speaker_label in matched_speakers or name in matched_profiles:
                continue
            if combined >= threshold:
                # Check for near-ties: is there another unmatched profile
                # within 0.05 of this score for the same speaker?
                if detail.get("ecapa") is not None and detail["layers"] >= 2:
                    rivals = [
                        (c, n, d) for c, _, n, d in scores
                        if _ == speaker_label and n != name
                        and n not in matched_profiles
                        and abs(c - combined) < 0.05
                        and d.get("ecapa") is not None
                    ]
                    if rivals:
                        # Use ECAPA score to break the tie
                        my_ecapa = detail["ecapa"]
                        for _, rival_name, rival_detail in rivals:
                            rival_ecapa = rival_detail["ecapa"]
                            if rival_ecapa > my_ecapa:
                                logger.info(
                                    "Tiebreak: %s → '%s' (ecapa=%.3f) beats '%s' (ecapa=%.3f)",
                                    speaker_label, rival_name, rival_ecapa, name, my_ecapa,
                                )
                                name = rival_name
                                detail = rival_detail
                                combined = rivals[0][0]
                                break

                matches[speaker_label] = name
                matched_speakers.add(speaker_label)
                matched_profiles.add(name)
                logger.info(
                    "Identified %s as '%s' (combined=%.3f, %s)",
                    speaker_label, name, combined, detail,
                )
            else:
                logger.debug(
                    "No match for %s → '%s' (combined=%.3f < threshold=%.2f, %s)",
                    speaker_label, name, combined, threshold, detail,
                )

        return matches

    def save(self, path: str | Path) -> None:
        """Save speaker profiles to a JSON file."""
        path = Path(path)
        data = {name: p.model_dump() for name, p in self.profiles.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"speakers": data}, f, indent=2)
        logger.info("Speaker DB saved to %s (%d profiles)", path, len(self.profiles))

    @classmethod
    def load(cls, path: str | Path) -> SpeakerDB:
        """Load speaker profiles from a JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        db = cls()
        for name, profile_data in data.get("speakers", {}).items():
            db.profiles[name] = SpeakerProfile.model_validate(profile_data)
        logger.info("Speaker DB loaded from %s (%d profiles)", path, len(db.profiles))
        return db

    @property
    def names(self) -> list[str]:
        """List of enrolled speaker names."""
        return list(self.profiles.keys())

    def __len__(self) -> int:
        return len(self.profiles)

    def __repr__(self) -> str:
        return f"SpeakerDB(speakers={self.names})"
