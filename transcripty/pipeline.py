"""High-level convenience pipeline combining transcription + diarization."""

from __future__ import annotations

import logging
from pathlib import Path

from transcripty.merge import merge
from transcripty.models import LabeledSegment
from transcripty.transcribe import transcribe

logger = logging.getLogger(__name__)


def transcribe_with_speakers(
    audio_path: str | Path,
    hf_token: str | None = None,
    num_speakers: int | None = None,
    speaker_db: object | None = None,
    **transcribe_kwargs,
) -> list[LabeledSegment]:
    """Transcribe audio and assign speaker labels in one call.

    Runs transcription, diarization, optional speaker identification,
    and merges the results.

    Args:
        audio_path: Path to the audio file.
        hf_token: HuggingFace token for diarization.
        num_speakers: Expected number of speakers (None for auto).
        speaker_db: Optional SpeakerDB instance for speaker identification.
        **transcribe_kwargs: Additional arguments passed to transcribe().

    Returns:
        List of LabeledSegment with text, timestamps, and speaker labels.
    """
    from transcripty.diarize import diarize

    result = transcribe(audio_path, **transcribe_kwargs)

    speakers = diarize(
        audio_path,
        hf_token=hf_token,
        num_speakers=num_speakers,
    )

    names = None
    if speaker_db is not None:
        names = speaker_db.identify(speakers)

    return merge(result.segments, speakers.segments, speaker_names=names)
