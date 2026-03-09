"""Compute device detection (CUDA -> MPS -> CPU).

This module provides a simple device detection interface.
For full hardware profiling, use transcripty.hardware.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """Detect the optimal compute device.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
    Result is cached via hardware module.
    """
    from transcripty.hardware import detect_hardware

    return detect_hardware().device


def reset_cache() -> None:
    """Reset the cached device (useful for testing)."""
    from transcripty.hardware import reset_cache as hw_reset

    hw_reset()
