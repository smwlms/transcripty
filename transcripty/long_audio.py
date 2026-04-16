"""Long-form audio transcription with chunking, language detection, and context carryover.

Automatically splits audio longer than a configurable threshold into
overlapping chunks, detects the language of each chunk, transcribes in
the correct language with context from the previous chunk, and merges
everything back into a single :class:`TranscriptionResult`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transcripty.chunking import cleanup_chunks, split_audio
from transcripty.models import Segment, TranscriptionResult, Word

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language detection on raw audio
# ---------------------------------------------------------------------------
def detect_audio_language(
    audio_path: str | Path,
    model_size: str | None = None,
) -> tuple[str, float]:
    """Detect the dominant language of an audio file using Whisper.

    Only analyses the first 30 seconds — very fast.

    Args:
        audio_path: Path to the audio file.
        model_size: Whisper model to use.  *None* → config default.

    Returns:
        ``(language_code, probability)`` e.g. ``("nl", 0.97)``.
    """

    from transcripty.audio import wav_audio
    from transcripty.config import get_config
    from transcripty.device import detect_device
    from transcripty.transcribe import _get_model

    cfg = get_config()
    model_size = model_size or cfg.model_size
    device = detect_device()
    whisper_device = "auto" if device == "mps" else device
    model = _get_model(model_size, cfg.compute_type, whisper_device, cfg.cpu_threads)

    with wav_audio(audio_path) as wav_path:
        _, info = model.transcribe(
            str(wav_path),
            language=None,
            beam_size=1,
            vad_filter=True,
            # Only need language info — skip full transcription
            without_timestamps=True,
        )
        # Consume the generator minimally (needed for info to be populated)
        # faster-whisper populates info.language before yielding segments
        return info.language, info.language_probability


# ---------------------------------------------------------------------------
# Overlap deduplication
# ---------------------------------------------------------------------------
def _deduplicate_overlap(
    prev_segments: list[Segment],
    curr_segments: list[Segment],
    overlap_start: float,
    overlap_end: float,
) -> list[Segment]:
    """Remove duplicate segments in the overlap zone between two chunks.

    Keeps segments from ``prev_segments`` that end before the overlap
    midpoint, and segments from ``curr_segments`` that start after it.
    Segments straddling the midpoint are kept from whichever chunk has
    the higher ``avg_logprob`` (= more confident transcription).
    """
    if not prev_segments or not curr_segments:
        return prev_segments + curr_segments

    midpoint = (overlap_start + overlap_end) / 2

    # Keep prev segments that are clearly before midpoint
    kept_prev = [s for s in prev_segments if s.end <= midpoint]

    # Keep curr segments that are clearly after midpoint
    kept_curr = [s for s in curr_segments if s.start >= midpoint]

    # Segments straddling the midpoint — pick the better one
    prev_straddling = [s for s in prev_segments if s.start < midpoint < s.end]
    curr_straddling = [s for s in curr_segments if s.start < midpoint < s.end]

    for ps in prev_straddling:
        # Check if there's a corresponding curr segment
        best = ps
        for cs in curr_straddling:
            if _segments_overlap(ps, cs):
                # Pick the one with higher confidence
                ps_conf = ps.avg_logprob if ps.avg_logprob is not None else -1.0
                cs_conf = cs.avg_logprob if cs.avg_logprob is not None else -1.0
                if cs_conf > ps_conf:
                    best = cs
                break
        if best not in kept_prev and best not in kept_curr:
            kept_prev.append(best)

    # Add any curr straddling that weren't matched
    for cs in curr_straddling:
        already_covered = any(_segments_overlap(cs, s) for s in kept_prev)
        if not already_covered and cs not in kept_curr:
            kept_curr.append(cs)

    return sorted(kept_prev + kept_curr, key=lambda s: s.start)


def _segments_overlap(a: Segment, b: Segment) -> bool:
    """Check if two segments overlap in time."""
    return a.start < b.end and b.start < a.end


# ---------------------------------------------------------------------------
# Segment post-processing
# ---------------------------------------------------------------------------
def filter_hallucinations(
    segments: list[Segment],
    min_duration: float = 0.1,
    max_no_speech_prob: float = 0.8,
) -> list[Segment]:
    """Remove segments that are likely hallucinations.

    Filters out segments that are very short or have high no-speech
    probability — common artifacts on long audio with silence gaps.
    """
    filtered = []
    removed = 0
    for seg in segments:
        duration = seg.end - seg.start
        if duration < min_duration:
            removed += 1
            continue
        if seg.no_speech_prob is not None and seg.no_speech_prob > max_no_speech_prob:
            removed += 1
            continue
        filtered.append(seg)

    if removed:
        logger.info("Filtered %d likely hallucination segments", removed)
    return filtered


def merge_consecutive_segments(
    segments: list[Segment],
    max_gap: float = 0.3,
    max_merged_duration: float = 30.0,
) -> list[Segment]:
    """Merge consecutive segments with small gaps between them.

    Useful after chunk boundary deduplication where segments may have
    been fragmented.  Only merges segments in the same language.
    """
    if not segments:
        return segments

    merged: list[Segment] = [segments[0].model_copy(deep=True)]

    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg.start - prev.end
        same_lang = prev.language == seg.language
        merged_duration = seg.end - prev.start

        if gap <= max_gap and same_lang and merged_duration <= max_merged_duration:
            # Merge into previous
            prev.text = f"{prev.text} {seg.text}"
            prev.end = seg.end
            prev.words.extend(seg.words)
            # Keep the more confident logprob
            if seg.avg_logprob is not None and prev.avg_logprob is not None:
                prev.avg_logprob = max(prev.avg_logprob, seg.avg_logprob)
            if seg.no_speech_prob is not None and prev.no_speech_prob is not None:
                prev.no_speech_prob = min(prev.no_speech_prob, seg.no_speech_prob)
        else:
            merged.append(seg.model_copy(deep=True))

    if len(merged) < len(segments):
        logger.info(
            "Merged %d → %d segments (gap ≤ %.1fs)",
            len(segments),
            len(merged),
            max_gap,
        )
    return merged


# ---------------------------------------------------------------------------
# Core: chunked transcription
# ---------------------------------------------------------------------------
def transcribe_long(
    audio_path: str | Path,
    chunk_minutes: float = 10.0,
    overlap_seconds: float = 5.0,
    context_words: int = 50,
    detect_language_per_chunk: bool = True,
    on_progress: Callable[[float, str], None] | None = None,
    **transcribe_kwargs: Any,
) -> TranscriptionResult:
    """Transcribe long audio with chunking, per-chunk language detection, and context carryover.

    Pipeline per chunk:
      1. Detect language (Whisper, first 30s — fast)
      2. Build ``initial_prompt`` from previous chunk's last words
      3. Transcribe with detected language + prompt
      4. Adjust timestamps to the original timeline
      5. Deduplicate overlap zone with previous chunk

    Args:
        audio_path: Path to the audio file.
        chunk_minutes: Target chunk length in minutes.
        overlap_seconds: Overlap between consecutive chunks.
        context_words: Number of trailing words from the previous chunk
            to pass as ``initial_prompt`` for continuity.
        detect_language_per_chunk: When *True*, detect the language of each
            chunk independently.  Crucial for multilingual meetings.
        on_progress: Optional progress callback ``(0.0–1.0, message)``.
        **transcribe_kwargs: Extra arguments forwarded to
            :func:`transcripty.transcribe.transcribe` (e.g. *model_size*,
            *vad_filter*, *beam_size*).

    Returns:
        A single :class:`TranscriptionResult` with all segments on the
        original timeline.
    """
    from transcripty.transcribe import transcribe

    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if on_progress:
        on_progress(0.0, "Splitting audio into chunks...")

    chunks = split_audio(path, chunk_minutes, overlap_seconds)
    logger.info("Processing %d chunks for %s", len(chunks), path.name)

    all_segments: list[Segment] = []
    global_language = "en"
    global_language_prob = 0.0
    total_duration = chunks[-1].end if chunks else 0.0
    prev_context = ""

    try:
        for i, chunk in enumerate(chunks):
            chunk_progress_base = i / len(chunks)
            chunk_progress_span = 1.0 / len(chunks)

            if on_progress:
                on_progress(
                    chunk_progress_base,
                    f"Chunk {i + 1}/{len(chunks)}: processing...",
                )

            # --- 1. Language detection ---
            chunk_lang = transcribe_kwargs.get("language")
            if detect_language_per_chunk and chunk_lang is None:
                try:
                    chunk_lang, lang_prob = detect_audio_language(
                        str(chunk.path),
                        model_size=transcribe_kwargs.get("model_size"),
                    )
                    logger.info("Chunk %d language: %s (%.2f)", i, chunk_lang, lang_prob)
                    if lang_prob > global_language_prob:
                        global_language = chunk_lang
                        global_language_prob = lang_prob
                except Exception as e:
                    logger.warning("Language detection failed for chunk %d: %s", i, e)

            # --- 2. Context carryover ---
            kwargs = dict(transcribe_kwargs)
            if chunk_lang:
                kwargs["language"] = chunk_lang
            if prev_context and "prompt" not in kwargs:
                kwargs["prompt"] = prev_context

            # Don't pass our custom params to transcribe()
            kwargs.pop("detect_language_per_chunk", None)
            kwargs.pop("chunk_minutes", None)
            kwargs.pop("overlap_seconds", None)
            kwargs.pop("context_words", None)
            kwargs.pop("auto_chunk", None)
            kwargs.pop("auto_chunk_threshold", None)

            # --- 3. Transcribe chunk ---
            result = transcribe(str(chunk.path), **kwargs)

            # --- 4. Adjust timestamps to original timeline ---
            chunk_segments = _shift_timestamps(result.segments, chunk.start)

            # --- 5. Deduplicate overlap with previous chunk ---
            if i > 0 and overlap_seconds > 0 and all_segments:
                overlap_start = chunk.start
                overlap_end = chunk.start + overlap_seconds
                all_segments = _deduplicate_overlap(
                    all_segments, chunk_segments, overlap_start, overlap_end
                )
            else:
                all_segments.extend(chunk_segments)

            # --- Update context for next chunk ---
            if context_words > 0 and chunk_segments:
                all_words = []
                for seg in chunk_segments:
                    all_words.extend(seg.text.split())
                prev_context = " ".join(all_words[-context_words:])

            if on_progress:
                on_progress(
                    chunk_progress_base + chunk_progress_span * 0.9,
                    f"Chunk {i + 1}/{len(chunks)}: done",
                )

    finally:
        cleanup_chunks(chunks)

    # --- Post-process ---
    all_segments = filter_hallucinations(all_segments)
    all_segments = merge_consecutive_segments(all_segments)

    if on_progress:
        on_progress(1.0, "Transcription complete")

    return TranscriptionResult(
        segments=all_segments,
        language=global_language,
        language_probability=global_language_prob,
        duration=total_duration,
    )


def _shift_timestamps(segments: list[Segment], offset: float) -> list[Segment]:
    """Create copies of segments with timestamps shifted by offset."""
    shifted = []
    for seg in segments:
        new_words = [
            Word(
                text=w.text,
                start=w.start + offset,
                end=w.end + offset,
                probability=w.probability,
                language=w.language,
            )
            for w in seg.words
        ]
        shifted.append(
            Segment(
                text=seg.text,
                start=seg.start + offset,
                end=seg.end + offset,
                words=new_words,
                language=seg.language,
                avg_logprob=seg.avg_logprob,
                no_speech_prob=seg.no_speech_prob,
            )
        )
    return shifted
