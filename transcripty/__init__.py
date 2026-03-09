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
from transcripty.speakers import SpeakerDB
from transcripty.transcribe import transcribe
from transcripty.vocabulary import Vocabulary

# Lazy import: diarize requires torch/pyannote which are optional
try:
    from transcripty.diarize import diarize
except ImportError:
    diarize = None  # type: ignore[assignment]

__all__ = [
    "transcribe",
    "diarize",
    "merge",
    "Vocabulary",
    "SpeakerDB",
    "TranscriptionResult",
    "DiarizationResult",
    "DiarizationSegment",
    "LabeledSegment",
    "Segment",
    "Word",
]
