"""Tests for device detection."""

from unittest.mock import MagicMock, patch

from transcripty.device import detect_device, reset_cache


def setup_function():
    """Reset device cache before each test."""
    reset_cache()


def test_detect_cpu_without_torch():
    with patch.dict("sys.modules", {"torch": None}):
        reset_cache()
        # When torch can't be imported, ImportError is raised inside detect_device
        # which falls back to "cpu"
        device = detect_device()
        assert device == "cpu"


def test_detect_cuda():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    with patch.dict("sys.modules", {"torch": mock_torch}):
        reset_cache()
        device = detect_device()
        assert device == "cuda"


def test_detect_mps():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    mock_torch.backends.mps.is_built.return_value = True
    with patch("platform.system", return_value="Darwin"):
        with patch.dict("sys.modules", {"torch": mock_torch}):
            reset_cache()
            device = detect_device()
            assert device == "mps"


def test_caching():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        reset_cache()
        d1 = detect_device()
        d2 = detect_device()
        assert d1 == d2 == "cpu"
        # torch.cuda.is_available should only be called once due to caching
        assert mock_torch.cuda.is_available.call_count == 1
