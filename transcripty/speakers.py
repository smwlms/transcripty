"""Speaker enrollment and identification via voice embeddings."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from transcripty.models import DiarizationResult

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)


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
        self.profiles: dict[str, dict] = {}
        # profiles = {
        #   "Name": {
        #     "embedding": [float, ...],
        #     "enrolled_at": "ISO timestamp"
        #   }
        # }

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

        self.profiles[name] = {
            "embedding": embedding,
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Enrolled '%s' (embedding dim=%d)", name, len(embedding))

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
        self.profiles[name] = {
            "embedding": embedding,
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Enrolled '%s' from embedding (dim=%d)", name, len(embedding))

    def identify(
        self,
        result: DiarizationResult,
        threshold: float = 0.5,
    ) -> dict[str, str]:
        """Identify speakers in a diarization result by matching embeddings.

        Compares each detected speaker's embedding against enrolled profiles
        using cosine similarity.

        Args:
            result: DiarizationResult with embeddings from diarize().
            threshold: Minimum cosine similarity to consider a match (0-1).

        Returns:
            Mapping of speaker labels to identified names.
            E.g. {"SPEAKER_00": "Alexander Willems", "SPEAKER_01": "Samuel"}
            Unmatched speakers are not included.
        """
        if not result.embeddings:
            logger.warning("No embeddings in diarization result. Cannot identify speakers.")
            return {}

        if not self.profiles:
            logger.warning("No enrolled speakers. Cannot identify.")
            return {}

        matches: dict[str, str] = {}

        for speaker_label, speaker_emb in result.embeddings.items():
            best_name = None
            best_score = -1.0

            for name, profile in self.profiles.items():
                score = _cosine_similarity(speaker_emb, profile["embedding"])
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_name and best_score >= threshold:
                matches[speaker_label] = best_name
                logger.info(
                    "Identified %s as '%s' (similarity=%.3f)",
                    speaker_label,
                    best_name,
                    best_score,
                )
            else:
                logger.info(
                    "No match for %s (best=%.3f, threshold=%.2f)",
                    speaker_label,
                    best_score,
                    threshold,
                )

        return matches

    def save(self, path: str | Path) -> None:
        """Save speaker profiles to a JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"speakers": self.profiles}, f, indent=2)
        logger.info("Speaker DB saved to %s (%d profiles)", path, len(self.profiles))

    @classmethod
    def load(cls, path: str | Path) -> SpeakerDB:
        """Load speaker profiles from a JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        db = cls()
        db.profiles = data.get("speakers", {})
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
