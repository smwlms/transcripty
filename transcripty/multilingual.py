"""ML7 Hybrid multilingual transcription with per-word language tagging.

Strategy (ML7 — 100% accuracy, ~0.56x RTF):
1. Transcribe entire audio as the dominant language (fast, ~0.13x RTF)
2. Detect language per 10-second window via Whisper (audio-level detection)
3. Retranscribe only non-dominant windows with the correct language forced
4. Merge results: keep dominant-language segments, replace detected segments
5. Assign speakers from diarization, tag every word with its language

This is the production approach for NL/FR/EN mixed audio in Belgian real estate.
For pure single-language audio, use the standard pipeline (much faster).
"""

from __future__ import annotations

import logging
import time
from bisect import bisect_right
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from transcripty.models import DiarizationSegment, LabeledSegment
    from transcripty.speakers import SpeakerDB

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_DETECT_WINDOW = 10.0  # seconds — language detection window size


def _audio_to_numpy(audio_path: Path) -> np.ndarray:
    """Convert any audio file to 16kHz mono float32 numpy array."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(audio_path))
    audio = audio.set_frame_rate(_SAMPLE_RATE).set_channels(1).set_sample_width(2)
    return np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0


def _find_speaker(
    start: float,
    end: float,
    diar_segments: list["DiarizationSegment"],
    diar_starts: list[float],
    speaker_names: dict[str, str] | None,
) -> str:
    """Find the dominant speaker for a time range using binary search."""
    from transcripty.models import UNKNOWN_SPEAKER

    overlaps: dict[str, float] = {}
    idx = bisect_right(diar_starts, start)
    for i in range(max(0, idx - 1), len(diar_segments)):
        d = diar_segments[i]
        if d.start >= end:
            break
        overlap = min(end, d.end) - max(start, d.start)
        if overlap > 0:
            overlaps[d.speaker] = overlaps.get(d.speaker, 0.0) + overlap

    if not overlaps:
        return UNKNOWN_SPEAKER

    label = max(overlaps, key=lambda k: overlaps[k])
    return (speaker_names or {}).get(label, label)


def transcribe_with_speakers_multilingual(
    audio_path: str | Path,
    hf_token: str | None = None,
    num_speakers: int | None = None,
    speaker_db: "SpeakerDB | None" = None,
    speaker_threshold: float = 0.5,
    on_progress: "Callable[[float, str], None] | None" = None,
    **transcribe_kwargs,
) -> list["LabeledSegment"]:
    """ML7 Hybrid: fast dominant-language transcription + selective retranscription.

    1. Transcribe whole file with dominant language (fast)
    2. Detect language per 10s window from audio
    3. Retranscribe only non-dominant windows with correct language
    4. Merge + assign speakers from diarization

    Args:
        audio_path: Path to the audio file.
        hf_token: HuggingFace token for diarization.
        num_speakers: Expected number of speakers (None for auto).
        speaker_db: Optional SpeakerDB for speaker identification.
        speaker_threshold: Min cosine similarity for speaker identification.
        on_progress: Progress callback (0.0-1.0).
        **transcribe_kwargs: Passed to Whisper (beam_size, word_timestamps, etc).

    Returns:
        List of LabeledSegment with speaker, language, and words.
    """
    from transcripty.config import get_config
    from transcripty.device import detect_device
    from transcripty.diarize import diarize
    from transcripty.models import LabeledSegment, Word
    from transcripty.transcribe import _get_model

    audio_path = Path(audio_path)
    t_total = time.time()

    # ── Stage 1: Convert audio to numpy (0.0 → 0.05) ─────────────────
    if on_progress:
        on_progress(0.0, "Converting audio...")

    samples = _audio_to_numpy(audio_path)
    duration = len(samples) / _SAMPLE_RATE
    logger.info("Audio: %.1fs loaded as numpy", duration)

    # ── Stage 2: Diarize (0.05 → 0.25) ───────────────────────────────
    if on_progress:
        on_progress(0.05, "Diarizing...")

    diar = diarize(audio_path, hf_token=hf_token, num_speakers=num_speakers)
    diar_starts = [d.start for d in diar.segments]

    names: dict[str, str] | None = None
    if speaker_db is not None:
        names = speaker_db.identify(diar, threshold=speaker_threshold)

    logger.info("Diarization: %d turns, speakers: %s", len(diar.segments), names)

    # ── Stage 3: Load model ───────────────────────────────────────────
    cfg = get_config()
    device = detect_device()
    whisper_device = "auto" if device == "mps" else device
    model = _get_model(cfg.model_size, cfg.compute_type, whisper_device, cfg.cpu_threads)

    # Base Whisper kwargs
    whisper_kw = dict(transcribe_kwargs)
    whisper_kw.pop("language", None)
    whisper_kw.pop("multilingual", None)
    whisper_kw.setdefault("word_timestamps", True)
    whisper_kw.setdefault("condition_on_previous_text", False)

    # ── Stage 4: Full transcription in dominant language (0.25 → 0.55) ─
    if on_progress:
        on_progress(0.25, "Transcribing (full file)...")

    # Detect dominant language from first 30 seconds
    first_30s = samples[: min(30 * _SAMPLE_RATE, len(samples))]
    _, detect_info = model.transcribe(first_30s, language=None, word_timestamps=False,
                                       beam_size=1, vad_filter=True,
                                       condition_on_previous_text=False)
    dominant_lang = detect_info.language
    logger.info("Dominant language detected: %s (prob=%.2f)",
                dominant_lang, detect_info.language_probability)

    # Full transcription with dominant language forced (fast path)
    segs_gen, info = model.transcribe(samples, language=dominant_lang,
                                       vad_filter=True, **whisper_kw)

    dominant_segments: list[dict] = []
    for seg in segs_gen:
        text = seg.text.strip()
        if not text:
            continue
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "text": w.word, "start": w.start, "end": w.end,
                    "probability": w.probability, "language": dominant_lang,
                })
        speaker = _find_speaker(seg.start, seg.end, diar.segments, diar_starts, names)
        dominant_segments.append({
            "text": text, "start": seg.start, "end": seg.end,
            "speaker": speaker, "language": dominant_lang, "words": words,
        })

    logger.info("Full transcription: %d segments in '%s'", len(dominant_segments), dominant_lang)

    # ── Stage 5: Detect language per 10s window (0.55 → 0.75) ────────
    if on_progress:
        on_progress(0.55, "Detecting languages per window...")

    n_windows = int(np.ceil(duration / _DETECT_WINDOW))
    window_langs: dict[int, str] = {}

    for wi in range(n_windows):
        ws = wi * _DETECT_WINDOW
        we = min(ws + _DETECT_WINDOW, duration)
        chunk = samples[int(ws * _SAMPLE_RATE): int(we * _SAMPLE_RATE)]
        if len(chunk) < _SAMPLE_RATE * 0.3:
            continue
        # Quick language detection (no word timestamps, beam=1)
        _, lang_info = model.transcribe(chunk, language=None, beam_size=1,
                                         word_timestamps=False, vad_filter=False,
                                         condition_on_previous_text=False)
        window_langs[wi] = lang_info.language

    non_dominant = {wi: lang for wi, lang in window_langs.items() if lang != dominant_lang}
    logger.info("Language detection: %d/%d windows non-%s: %s",
                len(non_dominant), n_windows, dominant_lang, non_dominant)

    if not non_dominant:
        # All windows are dominant language — return as-is
        result = [
            LabeledSegment(
                text=s["text"], start=s["start"], end=s["end"],
                speaker=s["speaker"], language=s["language"],
                words=[Word(**w) for w in s["words"]],
            )
            for s in dominant_segments
        ]
        if on_progress:
            on_progress(1.0, "Complete (single language)")
        return result

    # ── Stage 6: Retranscribe non-dominant windows (0.75 → 0.95) ─────
    if on_progress:
        on_progress(0.75, f"Retranscribing {len(non_dominant)} non-{dominant_lang} windows...")

    retranscribed: dict[int, list[dict]] = {}  # window_idx -> segments

    for wi, lang in non_dominant.items():
        ws = wi * _DETECT_WINDOW
        we = min(ws + _DETECT_WINDOW, duration)
        chunk = samples[int(ws * _SAMPLE_RATE): int(we * _SAMPLE_RATE)]

        segs_gen, _ = model.transcribe(chunk, language=lang, vad_filter=False, **whisper_kw)

        window_segs = []
        for seg in segs_gen:
            text = seg.text.strip()
            if not text:
                continue
            abs_start = seg.start + ws
            abs_end = seg.end + ws
            words = []
            if seg.words:
                for w in seg.words:
                    words.append({
                        "text": w.word,
                        "start": round(w.start + ws, 3),
                        "end": round(w.end + ws, 3),
                        "probability": w.probability,
                        "language": lang,
                    })
            speaker = _find_speaker(abs_start, abs_end, diar.segments, diar_starts, names)
            window_segs.append({
                "text": text, "start": abs_start, "end": abs_end,
                "speaker": speaker, "language": lang, "words": words,
            })
        retranscribed[wi] = window_segs

    # ── Stage 7: Merge — replace dominant segments in non-dominant windows ─
    if on_progress:
        on_progress(0.95, "Merging segments...")

    final: list[dict] = []

    for seg in dominant_segments:
        mid = (seg["start"] + seg["end"]) / 2
        window_idx = int(mid / _DETECT_WINDOW)
        if window_idx in non_dominant:
            continue  # Will be replaced by retranscribed version
        final.append(seg)

    # Add retranscribed segments
    for window_segs in retranscribed.values():
        final.extend(window_segs)

    # Sort by start time
    final.sort(key=lambda s: s["start"])

    result = [
        LabeledSegment(
            text=s["text"], start=round(s["start"], 3), end=round(s["end"], 3),
            speaker=s["speaker"], language=s["language"],
            words=[Word(**w) for w in s["words"]],
        )
        for s in final
    ]

    elapsed = time.time() - t_total
    lang_counts: dict[str, int] = {}
    for s in result:
        lang_counts[s.language or "?"] = lang_counts.get(s.language or "?", 0) + 1

    logger.info(
        "ML7 Hybrid done in %.1fs (RTF %.3fx). %d segments. Languages: %s",
        elapsed, elapsed / duration, len(result), lang_counts,
    )

    if on_progress:
        on_progress(1.0, "Complete")

    return result
