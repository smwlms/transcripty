"""Transcripty - Standalone audio transcription and diarization package."""

from transcripty.merge import merge
from transcripty.models import (
    DiarizationResult,
    DiarizationSegment,
    LabeledSegment,
    Segment,
    TranscriptionResult,
    Word,
)
from transcripty.transcribe import transcribe

__all__ = [
    "transcribe",
    "diarize",
    "merge",
    "TranscriptionResult",
    "DiarizationResult",
    "DiarizationSegment",
    "LabeledSegment",
    "Segment",
    "Word",
]


def __getattr__(name: str):
    """Lazy import for diarize to avoid requiring torch at import time."""
    if name == "diarize":
        from transcripty.diarize import diarize

        return diarize
    raise AttributeError(f"module 'transcripty' has no attribute {name!r}")
