"""Whisper transcription via faster-whisper."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from transcripty.audio import wav_audio
from transcripty.cache import ModelCache
from transcripty.config import get_config
from transcripty.device import detect_device
from transcripty.models import Segment, TranscriptionResult, Word

logger = logging.getLogger(__name__)

_UNSET = object()  # sentinel to distinguish "not provided" from None

ModelSize = Literal[
    "tiny", "base", "small", "medium",
    "large-v3", "large-v3-turbo", "distil-large-v3",
]
ComputeType = Literal["int8", "float16", "float32", "auto"]

_model_cache = ModelCache("whisper model")


def _get_model(model_size: str, compute_type: str, device: str, cpu_threads: int = 0):
    """Get or create a cached WhisperModel instance (thread-safe)."""
    from faster_whisper import WhisperModel

    cache_key = f"{model_size}:{compute_type}:{device}"
    _model_cache.max_size = get_config().max_cached_models

    kwargs: dict = {"device": device, "compute_type": compute_type}
    if cpu_threads > 0:
        kwargs["cpu_threads"] = cpu_threads

    return _model_cache.get_or_load(
        cache_key,
        lambda: WhisperModel(model_size, **kwargs),
    )


def clear_model_cache() -> None:
    """Clear the Whisper model cache."""
    _model_cache.clear()


def transcribe(
    audio_path: str | Path,
    model_size: ModelSize | None = None,
    language: str | None = None,
    multilingual: bool | None = None,
    word_timestamps: bool | None = None,
    compute_type: ComputeType | None = None,
    beam_size: int | None = None,
    prompt: str | None = None,
    vad_filter: bool | None = None,
    condition_on_previous_text: bool | None = None,
    hallucination_silence_threshold: float | None = _UNSET,
    repetition_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
    temperature: float | list[float] | tuple[float, ...] | None = None,
    chunk_length: int | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> TranscriptionResult:
    """Transcribe an audio file using faster-whisper.

    Args:
        audio_path: Path to the audio file (any format supported by pydub/ffmpeg).
        model_size: Whisper model size. Defaults to config value.
        language: Language code (e.g. "nl", "en"). None for auto-detection.
        multilingual: Perform language detection on every 30-second chunk.
            When True + language=None: per-chunk auto-detection (ideal for mixed-language audio).
            When True + language="nl": forces "nl" but still detects per chunk.
            When False (default): detect once in the first 30 seconds.
            Note: faster-whisper does not expose detected language per segment,
            so segment.language will contain the globally detected language.
        word_timestamps: Whether to include word-level timestamps. Defaults to config.
        compute_type: Quantization type. Defaults to config value.
        beam_size: Beam size for decoding. Defaults to config value.
        prompt: Initial prompt to bias recognition toward specific words/phrases.
        vad_filter: Enable Silero VAD to filter non-speech audio. Reduces hallucinations.
        condition_on_previous_text: Use previous output as prompt for next segment.
            Set to False to reduce hallucination cascades on long audio.
        hallucination_silence_threshold: Skip segments generated after this many
            seconds of silence (requires word_timestamps=True).
        repetition_penalty: Penalize repeated tokens (>1.0 reduces repetitions).
        no_repeat_ngram_size: Prevent repetition of n-grams of this size.
        temperature: Sampling temperature. 0.0 = greedy/deterministic decoding.
            Values >0 (e.g. 0.2) add randomness and can reduce repetitive hallucinations
            on silent or low-speech audio. Defaults to config value (0.0).
        on_progress: Optional callback ``(progress: float, message: str) -> None``.
            Progress is 0.0–1.0 based on segment end time vs audio duration.

    Returns:
        TranscriptionResult with segments, detected language, and duration.
    """
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "faster-whisper is required for transcription. "
            "Install with: pip install faster-whisper"
        ) from e

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Resolve defaults from config
    cfg = get_config()
    model_size = model_size or cfg.model_size  # type: ignore[assignment]
    compute_type = compute_type or cfg.compute_type  # type: ignore[assignment]
    beam_size = beam_size if beam_size is not None else cfg.beam_size
    word_timestamps = word_timestamps if word_timestamps is not None else cfg.word_timestamps
    language = language if language is not None else cfg.language
    multilingual = multilingual if multilingual is not None else cfg.multilingual
    vad_filter = vad_filter if vad_filter is not None else cfg.vad_filter
    condition_on_previous_text = (
        condition_on_previous_text
        if condition_on_previous_text is not None
        else cfg.condition_on_previous_text
    )
    repetition_penalty = (
        repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty
    )
    no_repeat_ngram_size = (
        no_repeat_ngram_size if no_repeat_ngram_size is not None else cfg.no_repeat_ngram_size
    )
    temperature = temperature if temperature is not None else cfg.temperature
    if hallucination_silence_threshold is _UNSET:
        hallucination_silence_threshold = cfg.hallucination_silence_threshold

    # Determine device for whisper
    device = detect_device()
    # CTranslate2 (used by faster-whisper) doesn't support MPS
    whisper_device = "auto" if device == "mps" else device

    model = _get_model(model_size, compute_type, whisper_device, cfg.cpu_threads)

    with wav_audio(audio_path) as wav_path:
        logger.info("Transcribing %s...", wav_path.name)
        start = time.time()

        transcribe_kwargs: dict = {
            "beam_size": beam_size,
            "language": language,
            "multilingual": multilingual,
            "word_timestamps": word_timestamps,
            "vad_filter": vad_filter,
            "condition_on_previous_text": condition_on_previous_text,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "temperature": temperature,
        }
        if chunk_length is not None:
            transcribe_kwargs["chunk_length"] = chunk_length
        if prompt:
            transcribe_kwargs["initial_prompt"] = prompt
            logger.info("Using custom prompt: %s", prompt[:80])
        if hallucination_silence_threshold is not None:
            transcribe_kwargs["hallucination_silence_threshold"] = (
                hallucination_silence_threshold
            )

        segments_gen, info = model.transcribe(str(wav_path), **transcribe_kwargs)

        if on_progress:
            on_progress(0.0, "Transcribing...")

        segments: list[Segment] = []
        for seg in segments_gen:
            words: list[Word] = []
            if word_timestamps and seg.words:
                words = [
                    Word(
                        text=w.word,
                        start=w.start,
                        end=w.end,
                        probability=w.probability,
                    )
                    for w in seg.words
                ]

            segments.append(
                Segment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    words=words,
                    language=info.language,
                    avg_logprob=getattr(seg, "avg_logprob", None),
                    no_speech_prob=getattr(seg, "no_speech_prob", None),
                )
            )

            if on_progress and info.duration > 0:
                progress = min(seg.end / info.duration, 1.0)
                on_progress(progress, "Transcribing...")

        # Post-process: per-segment language detection via lingua-py
        if multilingual and segments:
            try:
                from transcripty.language_detect import detect_segment_languages

                segments = detect_segment_languages(segments)
            except Exception as e:
                logger.warning("Per-segment language detection failed: %s", e)

        if on_progress:
            on_progress(1.0, "Transcription complete")

        elapsed = round(time.time() - start, 2)
        logger.info(
            "Transcription complete in %ss. %d segments, language=%s (%.2f)",
            elapsed,
            len(segments),
            info.language,
            info.language_probability,
        )

        return TranscriptionResult(
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )
