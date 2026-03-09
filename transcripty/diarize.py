"""Speaker diarization via pyannote.audio."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from transcripty.audio import wav_audio
from transcripty.device import detect_device
from transcripty.models import DiarizationResult, DiarizationSegment

logger = logging.getLogger(__name__)

DEFAULT_PIPELINE = "pyannote/speaker-diarization-3.1"


def _resolve_hf_token(hf_token: str | None) -> str | None:
    """Resolve HuggingFace token: parameter > HF_TOKEN env var."""
    if hf_token:
        return hf_token

    # Try loading .env if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    return os.environ.get("HF_TOKEN")


def diarize(
    audio_path: str | Path,
    hf_token: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int = 1,
    max_speakers: int = 10,
    pipeline: str = DEFAULT_PIPELINE,
) -> DiarizationResult:
    """Run speaker diarization on an audio file using pyannote.

    Args:
        audio_path: Path to the audio file.
        hf_token: HuggingFace API token. Falls back to HF_TOKEN env var / .env.
        num_speakers: Exact number of speakers (None for auto-detection).
        min_speakers: Minimum expected speakers (used when num_speakers is None).
        max_speakers: Maximum expected speakers (used when num_speakers is None).
        pipeline: Pyannote pipeline model name.

    Returns:
        DiarizationResult with speaker segments and detected speaker count.
    """
    try:
        import torch
        from pyannote.audio import Pipeline as PyannotePipeline
    except ImportError as e:
        raise ImportError(
            "pyannote.audio and torch are required for diarization. "
            "Install with: pip install transcripty[diarization]"
        ) from e

    audio_path = Path(audio_path)
    token = _resolve_hf_token(hf_token)

    if not token:
        logger.warning(
            "No HuggingFace token provided. Set HF_TOKEN in .env or pass hf_token parameter."
        )

    # Load pipeline
    device = detect_device()
    logger.info("Loading pyannote pipeline '%s' on %s...", pipeline, device)

    auth_kwargs = {"use_auth_token": token} if token else {}
    diarization_pipeline = PyannotePipeline.from_pretrained(pipeline, **auth_kwargs)
    diarization_pipeline.to(torch.device(device))

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

        annotation = diarization_pipeline(str(wav_path), **params)

        elapsed = round(time.time() - start, 2)
        speakers = annotation.labels()
        logger.info(
            "Diarization complete in %ss. %d speakers detected.",
            elapsed,
            len(speakers),
        )

        # Convert pyannote Annotation to our models
        segments: list[DiarizationSegment] = []
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    speaker=speaker,
                    start=segment.start,
                    end=segment.end,
                )
            )

        return DiarizationResult(
            segments=segments,
            num_speakers=len(speakers),
        )
