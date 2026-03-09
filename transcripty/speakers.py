"""Speaker enrollment and identification via voice embeddings."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from transcripty.models import DiarizationResult

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
    enrolled_at: str


class SpeakerDB:
    """Database of known speaker voice profiles (embeddings).

    Supports enrolling speakers from reference audio and identifying
    speakers in diarization results by comparing embeddings.

    Usage:
        db = SpeakerDB()
        db.enroll("Alexander Willems", "alex_reference.mp3")
        db.save("speakers.json")

        db = SpeakerDB.load("speakers.json")
        names = db.identify(diarization_result)
        # {"SPEAKER_00": "Alexander Willems"}
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
    ) -> None:
        """Enroll a speaker directly from a pre-computed embedding.

        Args:
            name: Display name for this speaker.
            embedding: Speaker embedding vector.
        """
        self.profiles[name] = SpeakerProfile(
            embedding=embedding,
            enrolled_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("Enrolled '%s' from embedding (dim=%d)", name, len(embedding))

    def identify(
        self,
        result: DiarizationResult,
        threshold: float = 0.5,
    ) -> dict[str, str]:
        """Identify speakers using exclusive greedy matching.

        Each speaker label is matched to at most one profile, and each
        profile is matched to at most one speaker label. Highest scoring
        pairs are matched first.

        Args:
            result: DiarizationResult with embeddings from diarize().
            threshold: Minimum cosine similarity to consider a match (0-1).

        Returns:
            Mapping of speaker labels to identified names.
        """
        if not result.embeddings:
            logger.warning("No embeddings in diarization result. Cannot identify speakers.")
            return {}

        if not self.profiles:
            logger.warning("No enrolled speakers. Cannot identify.")
            return {}

        # Compute all pairwise scores
        scores: list[tuple[float, str, str]] = []
        for speaker_label, speaker_emb in result.embeddings.items():
            for name, profile in self.profiles.items():
                score = _cosine_similarity(speaker_emb, profile.embedding)
                scores.append((score, speaker_label, name))

        # Greedy exclusive matching: highest score first
        scores.sort(reverse=True)
        matched_speakers: set[str] = set()
        matched_profiles: set[str] = set()
        matches: dict[str, str] = {}

        for score, speaker_label, name in scores:
            if speaker_label in matched_speakers or name in matched_profiles:
                continue
            if score >= threshold:
                matches[speaker_label] = name
                matched_speakers.add(speaker_label)
                matched_profiles.add(name)
                logger.info(
                    "Identified %s as '%s' (similarity=%.3f)",
                    speaker_label,
                    name,
                    score,
                )
            else:
                logger.info(
                    "No match for %s → '%s' (score=%.3f < threshold=%.2f)",
                    speaker_label,
                    name,
                    score,
                    threshold,
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
