"""Output formatters for transcription results (SRT, VTT, plain text)."""

from __future__ import annotations

from transcripty.models import LabeledSegment, Segment, WordHighlight


def _fmt_srt_ts(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt_ts(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def to_srt(segments: list[Segment] | list[LabeledSegment]) -> str:
    """Convert segments to SRT subtitle format.

    Args:
        segments: List of Segment or LabeledSegment.

    Returns:
        SRT formatted string.
    """
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        start = _fmt_srt_ts(seg.start)
        end = _fmt_srt_ts(seg.end)
        text = seg.text
        if hasattr(seg, "speaker") and seg.speaker:
            text = f"[{seg.speaker}] {text}"
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def to_vtt(segments: list[Segment] | list[LabeledSegment]) -> str:
    """Convert segments to WebVTT subtitle format.

    Args:
        segments: List of Segment or LabeledSegment.

    Returns:
        WebVTT formatted string.
    """
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        start = _fmt_vtt_ts(seg.start)
        end = _fmt_vtt_ts(seg.end)
        text = seg.text
        if hasattr(seg, "speaker") and seg.speaker:
            text = f"<v {seg.speaker}>{text}"
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def to_text(
    segments: list[Segment] | list[LabeledSegment],
    include_speakers: bool = True,
    include_timestamps: bool = False,
) -> str:
    """Convert segments to plain text.

    Args:
        segments: List of Segment or LabeledSegment.
        include_speakers: Prefix lines with speaker name (if available).
        include_timestamps: Include timestamps before each line.

    Returns:
        Plain text string.
    """
    lines: list[str] = []
    for seg in segments:
        parts: list[str] = []

        if include_timestamps:
            m = int(seg.start // 60)
            s = int(seg.start % 60)
            parts.append(f"[{m:02d}:{s:02d}]")

        if include_speakers and hasattr(seg, "speaker") and seg.speaker:
            parts.append(f"{seg.speaker}:")

        parts.append(seg.text)
        lines.append(" ".join(parts))

    return "\n".join(lines)


def to_word_highlights(
    segments: list[Segment] | list[LabeledSegment],
) -> list[WordHighlight]:
    """Extract a flat list of word-level timing data for frontend highlighting.

    Flattens all words from all segments into a single chronological list.
    Each word carries its segment index and optional speaker label, so a
    frontend player can highlight words in sync with audio playback.

    Requires that transcription was run with ``word_timestamps=True``.
    Segments without words are skipped.

    Args:
        segments: List of Segment or LabeledSegment (with word timestamps).

    Returns:
        Flat list of WordHighlight, ordered chronologically.
    """
    highlights: list[WordHighlight] = []
    for i, seg in enumerate(segments):
        speaker = seg.speaker if hasattr(seg, "speaker") else None
        for w in seg.words:
            highlights.append(
                WordHighlight(
                    word=w.text,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                    segment_index=i,
                    speaker=speaker,
                )
            )
    return highlights
