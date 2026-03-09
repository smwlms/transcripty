"""Transcripty - Standalone audio transcription and diarization package."""

from importlib.metadata import PackageNotFoundError, version

from transcripty.config import configure, get_config
from transcripty.formatters import to_srt, to_text, to_vtt, to_word_highlights
from transcripty.merge import merge
from transcripty.models import (
    UNKNOWN_SPEAKER,
    DiarizationResult,
    DiarizationSegment,
    LabeledSegment,
    Segment,
    TranscriptionResult,
    Word,
    WordHighlight,
)
from transcripty.transcribe import transcribe
from transcripty.vocabulary import Vocabulary

# Lazy imports: these require optional dependencies (torch/pyannote/numpy)
try:
    from transcripty.diarize import diarize
except ImportError:
    diarize = None  # type: ignore[assignment]

try:
    from transcripty.pipeline import transcribe_with_speakers
except ImportError:
    transcribe_with_speakers = None  # type: ignore[assignment]

try:
    from transcripty.speakers import SpeakerDB
except ImportError:
    SpeakerDB = None  # type: ignore[assignment]

try:
    __version__ = version("transcripty")
except PackageNotFoundError:
    __version__ = "0.2.0"


def clear_cache() -> None:
    """Clear all model/pipeline caches."""
    from transcripty.transcribe import clear_model_cache

    clear_model_cache()
    try:
        from transcripty.diarize import clear_pipeline_cache

        clear_pipeline_cache()
    except ImportError:
        pass


__all__ = [
    "__version__",
    "clear_cache",
    "configure",
    "diarize",
    "get_config",
    "merge",
    "to_srt",
    "to_text",
    "to_vtt",
    "to_word_highlights",
    "transcribe",
    "transcribe_with_speakers",
    "Vocabulary",
    "SpeakerDB",
    "TranscriptionResult",
    "DiarizationResult",
    "DiarizationSegment",
    "LabeledSegment",
    "Segment",
    "Word",
    "WordHighlight",
    "UNKNOWN_SPEAKER",
]
