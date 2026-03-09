"""Tests for configuration system."""

import pytest

from transcripty.config import (
    TranscriptyConfig,
    configure,
    get_config,
    reset_config,
)


def test_default_config():
    cfg = get_config()
    assert cfg.model_size in ("tiny", "base", "small", "medium", "large-v3", "distil-large-v3")
    assert cfg.max_cached_models == 2
    assert cfg.num_workers >= 1


def test_configure_overrides():
    configure(model_size="tiny", beam_size=3)
    cfg = get_config()
    assert cfg.model_size == "tiny"
    assert cfg.beam_size == 3


def test_configure_preserves_existing():
    configure(model_size="medium")
    configure(beam_size=10)
    cfg = get_config()
    assert cfg.model_size == "medium"
    assert cfg.beam_size == 10


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTY_MODEL_SIZE", "base")
    # Direct TranscriptyConfig picks up env var
    cfg = TranscriptyConfig()
    assert cfg.model_size == "base"


def test_env_var_hf_token(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTY_HF_TOKEN", "test-token")
    reset_config()
    cfg = get_config()
    assert cfg.hf_token == "test-token"


def test_config_singleton():
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2


def test_reset_config():
    configure(model_size="tiny")
    reset_config()
    cfg = get_config()
    # Should be back to defaults (hardware-suggested)
    assert cfg.model_size != "tiny"


def test_yaml_config(tmp_path, monkeypatch):
    """Test loading from YAML file."""
    yaml_content = "model_size: medium\nbeam_size: 3\n"
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml_content)

    # Point config to temp file
    import transcripty.config as cfg_module

    original = cfg_module._DEFAULT_CONFIG_FILE
    cfg_module._DEFAULT_CONFIG_FILE = yaml_file
    TranscriptyConfig._yaml_file = str(yaml_file)
    reset_config()

    try:
        cfg = get_config()
        assert cfg.model_size == "medium"
        assert cfg.beam_size == 3
    finally:
        cfg_module._DEFAULT_CONFIG_FILE = original
        TranscriptyConfig._yaml_file = str(original)
        reset_config()


def test_configure_language_none_resets():
    """configure(language=None) should reset language to auto-detect."""
    configure(language="nl")
    cfg = get_config()
    assert cfg.language == "nl"

    configure(language=None)
    cfg = get_config()
    assert cfg.language is None


def test_max_cached_models_validation():
    with pytest.raises(Exception):
        TranscriptyConfig(max_cached_models=0)
