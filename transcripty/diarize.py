"""Speaker diarization via pyannote.audio."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from transcripty.audio import wav_audio
from transcripty.config import get_config
from transcripty.device import detect_device
from transcripty.models import DiarizationResult, DiarizationSegment

logger = logging.getLogger(__name__)

DEFAULT_PIPELINE = "pyannote/speaker-diarization-3.1"

_pipeline_lock = threading.Lock()
_pipeline_cache: dict[str, object] = {}

_cached_hf_token: str | None | bool = False  # False = not yet resolved


def _resolve_hf_token(hf_token: str | None) -> str | None:
    """Resolve HuggingFace token: parameter > config > HF_TOKEN env var."""
    if hf_token:
        return hf_token

    # Check config
    cfg_token = get_config().hf_token
    if cfg_token:
        return cfg_token

    # Cache env var lookup
    global _cached_hf_token
    if _cached_hf_token is not False:
        return _cached_hf_token  # type: ignore[return-value]

    # Try loading .env if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    _cached_hf_token = os.environ.get("HF_TOKEN")
    return _cached_hf_token  # type: ignore[return-value]


def _get_pipeline(pipeline_name: str, token: str | None, device: str):
    """Get or create a cached pyannote pipeline instance (thread-safe)."""
    import torch
    from pyannote.audio import Pipeline as PyannotePipeline

    cache_key = f"{pipeline_name}:{device}"
    with _pipeline_lock:
        if cache_key not in _pipeline_cache:
            # Evict oldest if cache is full
            max_cached = get_config().max_cached_models
            if len(_pipeline_cache) >= max_cached:
                oldest_key = next(iter(_pipeline_cache))
                logger.info("Evicting cached pipeline '%s'", oldest_key)
                del _pipeline_cache[oldest_key]

            logger.info("Loading pyannote pipeline '%s' on %s...", pipeline_name, device)
            p = PyannotePipeline.from_pretrained(pipeline_name, token=token)
            p.to(torch.device(device))
            _pipeline_cache[cache_key] = p
        else:
            logger.debug("Using cached pyannote pipeline '%s'", pipeline_name)
        return _pipeline_cache[cache_key]


def clear_pipeline_cache() -> None:
    """Clear the pyannote pipeline cache."""
    with _pipeline_lock:
        _pipeline_cache.clear()
    logger.info("Pyannote pipeline cache cleared")


def reset_hf_token_cache() -> None:
    """Reset the cached HF token (for testing)."""
    global _cached_hf_token
    _cached_hf_token = False


def diarize(
    audio_path: str | Path,
    hf_token: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    pipeline: str = DEFAULT_PIPELINE,
) -> DiarizationResult:
    """Run speaker diarization on an audio file using pyannote.

    Args:
        audio_path: Path to the audio file.
        hf_token: HuggingFace API token. Falls back to config/HF_TOKEN env var.
        num_speakers: Exact number of speakers (None for auto-detection).
        min_speakers: Minimum expected speakers. Defaults to config value.
        max_speakers: Maximum expected speakers. Defaults to config value.
        pipeline: Pyannote pipeline model name.

    Returns:
        DiarizationResult with speaker segments, speaker count, and embeddings.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        ValueError: If no HuggingFace token is available.
    """
    try:
        import torch  # noqa: F401
        from pyannote.audio import Pipeline as PyannotePipeline  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pyannote.audio and torch are required for diarization. "
            "Install with: pip install transcripty[diarization]"
        ) from e

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    token = _resolve_hf_token(hf_token)

    if not token:
        raise ValueError(
            "No HuggingFace token provided. Set HF_TOKEN in .env, "
            "use TRANSCRIPTY_HF_TOKEN env var, or pass hf_token parameter."
        )

    # Resolve defaults from config
    cfg = get_config()
    min_speakers = min_speakers if min_speakers is not None else cfg.min_speakers
    max_speakers = max_speakers if max_speakers is not None else cfg.max_speakers

    device = detect_device()
    diarization_pipeline = _get_pipeline(pipeline, token, device)

    # Run diarization
    with wav_audio(audio_path) as wav_path:
        logger.info("Diarizing %s...", wav_path.name)
        start = time.time()

        params: dict = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        else:
            params["min_speakers"] = min_speakers
            params["max_speakers"] = max_speakers

        output = diarization_pipeline(str(wav_path), **params)

        elapsed = round(time.time() - start, 2)

        # pyannote v4 returns DiarizeOutput dataclass, v3 returns Annotation
        if hasattr(output, "speaker_diarization"):
            annotation = output.speaker_diarization
        else:
            annotation = output

        # Convert pyannote Annotation to our models
        segments: list[DiarizationSegment] = []
        speaker_set: set[str] = set()
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            speaker_set.add(speaker)
            segments.append(
                DiarizationSegment(
                    speaker=speaker,
                    start=segment.start,
                    end=segment.end,
                )
            )

        # Extract speaker embeddings if available (pyannote v4)
        embeddings: dict[str, list[float]] = {}
        if hasattr(output, "speaker_embeddings") and output.speaker_embeddings is not None:
            speaker_labels = sorted(speaker_set)
            for i, label in enumerate(speaker_labels):
                if i < len(output.speaker_embeddings):
                    embeddings[label] = output.speaker_embeddings[i].tolist()
            logger.info(
                "Extracted embeddings for %d speakers (dim=%d)",
                len(embeddings),
                len(next(iter(embeddings.values()), [])),
            )

        logger.info(
            "Diarization complete in %ss. %d speakers detected.",
            elapsed,
            len(speaker_set),
        )

        return DiarizationResult(
            segments=segments,
            num_speakers=len(speaker_set),
            embeddings=embeddings,
        )
