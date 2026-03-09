"""Whisper transcription via faster-whisper."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from transcripty.audio import wav_audio
from transcripty.device import detect_device
from transcripty.models import Segment, TranscriptionResult, Word

logger = logging.getLogger(__name__)


def transcribe(
    audio_path: str | Path,
    model_size: str = "small",
    language: str | None = None,
    word_timestamps: bool = True,
    compute_type: str = "int8",
    beam_size: int = 5,
) -> TranscriptionResult:
    """Transcribe an audio file using faster-whisper.

    Args:
        audio_path: Path to the audio file (any format supported by pydub/ffmpeg).
        model_size: Whisper model size (tiny/base/small/medium/large-v3).
        language: Language code (e.g. "nl", "en"). None for auto-detection.
        word_timestamps: Whether to include word-level timestamps.
        compute_type: Quantization type (int8/float16/float32).
        beam_size: Beam size for decoding.

    Returns:
        TranscriptionResult with segments, detected language, and duration.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ImportError(
            "faster-whisper is required for transcription. "
            "Install with: pip install faster-whisper"
        ) from e

    audio_path = Path(audio_path)

    # Determine device for whisper
    device = detect_device()
    # CTranslate2 (used by faster-whisper) doesn't support MPS
    whisper_device = "auto" if device == "mps" else device

    logger.info(
        "Loading Whisper model '%s' (compute=%s, device=%s)...",
        model_size,
        compute_type,
        whisper_device,
    )
    model = WhisperModel(model_size, device=whisper_device, compute_type=compute_type)

    with wav_audio(audio_path) as wav_path:
        logger.info("Transcribing %s...", wav_path.name)
        start = time.time()

        segments_gen, info = model.transcribe(
            str(wav_path),
            beam_size=beam_size,
            language=language,
            word_timestamps=word_timestamps,
        )

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
