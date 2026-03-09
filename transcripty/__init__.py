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
from transcripty.vocabulary import Vocabulary

# Lazy imports: these require optional dependencies (torch/pyannote/numpy)
try:
    from transcripty.diarize import diarize
except ImportError:
    diarize = None  # type: ignore[assignment]

try:
    from transcripty.speakers import SpeakerDB
except ImportError:
    SpeakerDB = None  # type: ignore[assignment]

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
