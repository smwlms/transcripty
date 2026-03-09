"""Whisper transcription via faster-whisper."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

from transcripty.audio import wav_audio
from transcripty.cache import ModelCache
from transcripty.config import get_config
from transcripty.device import detect_device
from transcripty.models import Segment, TranscriptionResult, Word

logger = logging.getLogger(__name__)

ModelSize = Literal["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]
ComputeType = Literal["int8", "float16", "float32", "auto"]

_model_cache = ModelCache("whisper model")


def _get_model(model_size: str, compute_type: str, device: str):
    """Get or create a cached WhisperModel instance (thread-safe)."""
    from faster_whisper import WhisperModel

    cache_key = f"{model_size}:{compute_type}:{device}"
    _model_cache.max_size = get_config().max_cached_models
    return _model_cache.get_or_load(
        cache_key,
        lambda: WhisperModel(model_size, device=device, compute_type=compute_type),
    )


def clear_model_cache() -> None:
    """Clear the Whisper model cache."""
    _model_cache.clear()


def transcribe(
    audio_path: str | Path,
    model_size: ModelSize | None = None,
    language: str | None = None,
    word_timestamps: bool | None = None,
    compute_type: ComputeType | None = None,
    beam_size: int | None = None,
    prompt: str | None = None,
) -> TranscriptionResult:
    """Transcribe an audio file using faster-whisper.

    Args:
        audio_path: Path to the audio file (any format supported by pydub/ffmpeg).
        model_size: Whisper model size. Defaults to config value.
        language: Language code (e.g. "nl", "en"). None for auto-detection.
        word_timestamps: Whether to include word-level timestamps. Defaults to config.
        compute_type: Quantization type. Defaults to config value.
        beam_size: Beam size for decoding. Defaults to config value.
        prompt: Initial prompt to bias recognition toward specific words/phrases.

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

    # Determine device for whisper
    device = detect_device()
    # CTranslate2 (used by faster-whisper) doesn't support MPS
    whisper_device = "auto" if device == "mps" else device

    model = _get_model(model_size, compute_type, whisper_device)

    with wav_audio(audio_path) as wav_path:
        logger.info("Transcribing %s...", wav_path.name)
        start = time.time()

        transcribe_kwargs: dict = {
            "beam_size": beam_size,
            "language": language,
            "word_timestamps": word_timestamps,
        }
        if prompt:
            transcribe_kwargs["initial_prompt"] = prompt
            logger.info("Using custom prompt: %s", prompt[:80])

        segments_gen, info = model.transcribe(str(wav_path), **transcribe_kwargs)

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
                )
            )

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
