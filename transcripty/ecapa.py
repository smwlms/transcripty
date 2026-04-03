"""ECAPA-TDNN speaker embedding extraction via SpeechBrain.

Provides more discriminative embeddings than pyannote's default,
especially for speakers with similar voices (e.g. twins).

Usage:
    from transcripty.ecapa import extract_ecapa_embedding
    embedding = extract_ecapa_embedding("audio.mp3")
    # Returns list[float] of 192 dimensions
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def _get_model():
    """Get or load the ECAPA-TDNN model (thread-safe, singleton)."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        from speechbrain.inference.speaker import EncoderClassifier

        logger.info("Loading ECAPA-TDNN model (first use)...")
        _model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},  # MPS not supported by SpeechBrain
        )
        logger.info("ECAPA-TDNN model loaded")
        return _model


def extract_ecapa_embedding(
    audio_path: str | Path,
    start_s: float | None = None,
    end_s: float | None = None,
) -> list[float]:
    """Extract a speaker embedding from audio using ECAPA-TDNN.

    Args:
        audio_path: Path to audio file (any format supported by torchaudio).
        start_s: Optional start time in seconds (for extracting from a segment).
        end_s: Optional end time in seconds.

    Returns:
        192-dimensional embedding as list[float].
    """
    import torchaudio

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_model()

    # Load audio
    waveform, sample_rate = torchaudio.load(str(audio_path))

    # Trim to segment if specified
    if start_s is not None or end_s is not None:
        start_sample = int((start_s or 0) * sample_rate)
        end_sample = int((end_s or waveform.shape[1] / sample_rate) * sample_rate)
        end_sample = min(end_sample, waveform.shape[1])
        waveform = waveform[:, start_sample:end_sample]

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed (ECAPA expects 16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Extract embedding
    embedding = model.encode_batch(waveform)
    # Shape: (1, 1, 192) → flatten to (192,)
    embedding_np = embedding.squeeze().detach().cpu().numpy()

    # L2 normalize
    norm = np.linalg.norm(embedding_np)
    if norm > 0:
        embedding_np = embedding_np / norm

    return embedding_np.tolist()


def extract_ecapa_embeddings_for_segments(
    audio_path: str | Path,
    speaker_segments: dict[str, list[tuple[float, float]]],
) -> dict[str, list[float]]:
    """Extract ECAPA-TDNN embeddings for each speaker from their segments.

    Takes the diarization output (speaker label → list of (start, end) times)
    and extracts a single averaged embedding per speaker.

    Args:
        audio_path: Path to the audio file.
        speaker_segments: Dict mapping speaker labels to list of (start, end) tuples.

    Returns:
        Dict mapping speaker labels to 192-dim embeddings.
    """
    import torchaudio
    import torch

    audio_path = Path(audio_path)
    model = _get_model()

    # Load full audio once
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    embeddings = {}

    for speaker_label, segments in speaker_segments.items():
        # Concatenate all segments for this speaker (max 60s total)
        chunks = []
        total_samples = 0
        max_samples = 60 * sample_rate  # cap at 60 seconds

        for start, end in segments:
            if total_samples >= max_samples:
                break
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            end_sample = min(end_sample, waveform.shape[1])
            chunk = waveform[:, start_sample:end_sample]
            chunks.append(chunk)
            total_samples += chunk.shape[1]

        if not chunks:
            continue

        combined = torch.cat(chunks, dim=1)
        # Trim to max_samples
        if combined.shape[1] > max_samples:
            combined = combined[:, :max_samples]

        # Extract embedding
        emb = model.encode_batch(combined)
        emb_np = emb.squeeze().detach().cpu().numpy()

        # L2 normalize
        norm = np.linalg.norm(emb_np)
        if norm > 0:
            emb_np = emb_np / norm

        embeddings[speaker_label] = emb_np.tolist()
        logger.debug(
            "ECAPA embedding for %s: dim=%d, from %d segments (%.1fs)",
            speaker_label, len(emb_np), len(segments),
            total_samples / sample_rate,
        )

    return embeddings


def clear_ecapa_model() -> None:
    """Clear the cached ECAPA model (for testing/memory management)."""
    global _model
    with _model_lock:
        _model = None
        logger.info("ECAPA model cache cleared")
