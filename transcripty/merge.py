"""Merge transcription segments with diarization results."""

from __future__ import annotations

import logging
from bisect import bisect_right

from transcripty.models import (
    UNKNOWN_SPEAKER,
    DiarizationSegment,
    LabeledSegment,
    Segment,
)

logger = logging.getLogger(__name__)


def _find_speaker(
    start: float,
    end: float,
    diarization: list[DiarizationSegment],
    starts: list[float],
) -> str:
    """Find the dominant speaker for a time range using binary search."""
    overlaps: dict[str, float] = {}

    # Find the insertion point — start searching from segments that could overlap
    idx = bisect_right(starts, start)
    # Step back to catch segments that started before but may still overlap
    search_start = max(0, idx - 1)

    for i in range(search_start, len(diarization)):
        d = diarization[i]
        # Early exit: if this segment starts after our end, no more overlaps possible
        if d.start >= end:
            break

        overlap_start = max(start, d.start)
        overlap_end = min(end, d.end)
        overlap = overlap_end - overlap_start

        if overlap > 0:
            overlaps[d.speaker] = overlaps.get(d.speaker, 0.0) + overlap

    if not overlaps:
        return UNKNOWN_SPEAKER

    return max(overlaps, key=lambda k: overlaps[k])


def merge(
    segments: list[Segment],
    diarization: list[DiarizationSegment],
    speaker_names: dict[str, str] | None = None,
) -> list[LabeledSegment]:
    """Merge transcription segments with diarization to assign speakers.

    For each transcription segment, finds the speaker with the most temporal
    overlap from the diarization results.

    Args:
        segments: Transcription segments from transcribe().
        diarization: Speaker segments from diarize().
        speaker_names: Optional mapping of speaker labels to display names.
            E.g. {"SPEAKER_00": "Alexander Willems"}. From SpeakerDB.identify().

    Returns:
        List of LabeledSegment with speaker assignments.
    """
    logger.info(
        "Merging %d segments with %d diarization entries...",
        len(segments),
        len(diarization),
    )

    names = speaker_names or {}

    # Pre-compute sorted starts for binary search
    starts = [d.start for d in diarization]

    labeled: list[LabeledSegment] = []
    for seg in segments:
        speaker_label = _find_speaker(seg.start, seg.end, diarization, starts)
        speaker = names.get(speaker_label, speaker_label)
        labeled.append(
            LabeledSegment(
                text=seg.text,
                start=seg.start,
                end=seg.end,
                speaker=speaker,
                words=seg.words,
            )
        )

    speakers_found = set(s.speaker for s in labeled)
    logger.info(
        "Merge complete. %d labeled segments, speakers: %s",
        len(labeled),
        speakers_found,
    )

    return labeled
