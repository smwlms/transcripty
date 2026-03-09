"""Shared test fixtures."""

from unittest.mock import MagicMock

import pytest

from transcripty.config import reset_config
from transcripty.hardware import reset_cache as hw_reset


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset config and hardware cache between tests."""
    reset_config()
    hw_reset()
    yield
    reset_config()
    hw_reset()


@pytest.fixture
def mock_whisper_segment():
    """Create a mock faster-whisper segment."""

    def _make(text="Hello", start=0.0, end=1.0, words=None):
        seg = MagicMock()
        seg.text = text
        seg.start = start
        seg.end = end
        seg.words = words or []
        return seg

    return _make


@pytest.fixture
def mock_transcribe_info():
    """Create a mock faster-whisper info object."""

    def _make(language="en", language_probability=0.95, duration=10.0):
        info = MagicMock()
        info.language = language
        info.language_probability = language_probability
        info.duration = duration
        return info

    return _make
