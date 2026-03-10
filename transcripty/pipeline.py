"""High-level convenience pipeline combining transcription + diarization."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from transcripty.merge import merge
from transcripty.models import LabeledSegment
from transcripty.transcribe import transcribe

if TYPE_CHECKING:
    from transcripty.speakers import SpeakerDB

logger = logging.getLogger(__name__)


def _make_stage_callback(
    on_progress: Callable[[float, str], None],
    start: float,
    end: float,
) -> Callable[[float, str], None]:
    """Create a callback that maps 0.0-1.0 progress to a sub-range."""
    span = end - start

    def callback(progress: float, message: str) -> None:
        on_progress(start + progress * span, message)

    return callback


def transcribe_with_speakers(
    audio_path: str | Path,
    hf_token: str | None = None,
    num_speakers: int | None = None,
    speaker_db: SpeakerDB | None = None,
    on_progress: Callable[[float, str], None] | None = None,
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
        on_progress: Optional callback ``(progress: float, message: str) -> None``.
            Progress is 0.0–1.0, weighted across stages:
            0.0–0.7 transcription, 0.7–0.9 diarization, 0.9–1.0 merge/identify.
        **transcribe_kwargs: Additional arguments passed to transcribe().

    Returns:
        List of LabeledSegment with text, timestamps, and speaker labels.
    """
    from transcripty.diarize import diarize

    # Build stage callbacks that map to weighted progress ranges
    transcribe_cb = None
    diarize_cb = None
    if on_progress:
        transcribe_cb = _make_stage_callback(on_progress, 0.0, 0.7)
        diarize_cb = _make_stage_callback(on_progress, 0.7, 0.9)

    result = transcribe(audio_path, on_progress=transcribe_cb, **transcribe_kwargs)

    speakers = diarize(
        audio_path,
        hf_token=hf_token,
        num_speakers=num_speakers,
        on_progress=diarize_cb,
    )

    if on_progress:
        on_progress(0.9, "Merging segments...")

    names = None
    if speaker_db is not None:
        names = speaker_db.identify(speakers)

    labeled = merge(result.segments, speakers.segments, speaker_names=names)

    if on_progress:
        on_progress(1.0, "Complete")

    return labeled
