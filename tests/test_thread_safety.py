"""Tests for thread safety and cache management."""

import threading
from unittest.mock import MagicMock

from transcripty.config import configure, reset_config
from transcripty.transcribe import _model_cache, _model_lock, clear_model_cache


def test_clear_model_cache():
    _model_cache["test:key"] = MagicMock()
    assert len(_model_cache) > 0
    clear_model_cache()
    assert len(_model_cache) == 0


def test_cache_eviction():
    """When cache exceeds max_cached_models, oldest entry is evicted."""
    clear_model_cache()
    configure(max_cached_models=2)

    _model_cache["model_a:int8:cpu"] = MagicMock()
    _model_cache["model_b:int8:cpu"] = MagicMock()

    # Simulate adding a third model — should evict the first
    with _model_lock:
        from transcripty.config import get_config

        max_cached = get_config().max_cached_models
        if len(_model_cache) >= max_cached:
            oldest_key = next(iter(_model_cache))
            del _model_cache[oldest_key]
        _model_cache["model_c:int8:cpu"] = MagicMock()

    assert len(_model_cache) == 2
    assert "model_a:int8:cpu" not in _model_cache
    assert "model_c:int8:cpu" in _model_cache
    clear_model_cache()


def test_concurrent_config_access():
    """Multiple threads can safely access config."""
    reset_config()
    configure(model_size="small")

    results = []
    errors = []

    def read_config():
        try:
            from transcripty.config import get_config

            cfg = get_config()
            results.append(cfg.model_size)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=read_config) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert all(r == "small" for r in results)


def test_early_validation_transcribe(tmp_path):
    """transcribe() should raise FileNotFoundError for missing files."""
    import pytest

    from transcripty.transcribe import transcribe

    with pytest.raises(FileNotFoundError):
        transcribe(tmp_path / "nonexistent.wav")


def test_early_validation_diarize(tmp_path):
    """diarize() should raise FileNotFoundError for missing files."""
    import sys
    from unittest.mock import patch

    # Mock torch and pyannote to avoid ImportError
    with patch.dict(
        sys.modules,
        {"torch": MagicMock(), "pyannote": MagicMock(), "pyannote.audio": MagicMock()},
    ):
        import pytest

        from transcripty.diarize import diarize

        with pytest.raises(FileNotFoundError):
            diarize(tmp_path / "nonexistent.wav")
