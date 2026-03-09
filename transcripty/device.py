"""Compute device detection (CUDA -> MPS -> CPU)."""

from __future__ import annotations

import logging
import platform

logger = logging.getLogger(__name__)

_cached_device: str | None = None


def detect_device() -> str:
    """Detect the optimal compute device.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
    Result is cached after first call.
    """
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            logger.info("CUDA (NVIDIA GPU) detected.")
        elif (
            platform.system() == "Darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            device = "mps"
            logger.info("Apple MPS detected.")
        else:
            logger.info("No GPU detected, using CPU.")
    except ImportError:
        logger.debug("torch not installed, defaulting to CPU.")

    _cached_device = device
    return device


def reset_cache() -> None:
    """Reset the cached device (useful for testing)."""
    global _cached_device
    _cached_device = None
