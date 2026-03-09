"""Configuration management via Pydantic Settings."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = Path.home() / ".transcripty"
_DEFAULT_CONFIG_FILE = _DEFAULT_CONFIG_DIR / "config.yaml"

_config: TranscriptyConfig | None = None
_config_lock = threading.Lock()


def _yaml_settings_source():
    """Create YAML settings source if pydantic-settings[yaml] is available."""
    try:
        from pydantic_settings import YamlConfigSettingsSource

        return YamlConfigSettingsSource
    except ImportError:
        return None


class TranscriptyConfig(BaseSettings):
    """Transcripty configuration.

    Priority (highest to lowest):
    1. Programmatic overrides via configure()
    2. Environment variables (TRANSCRIPTY_*)
    3. YAML config file (~/.transcripty/config.yaml)
    4. Defaults (optionally from hardware profiling)
    """

    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIPTY_",
        extra="ignore",
    )

    _yaml_file: ClassVar[str] = str(_DEFAULT_CONFIG_FILE)

    # Transcription
    model_size: str = "small"
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 5
    word_timestamps: bool = True

    # Diarization
    hf_token: str | None = None
    num_speakers: int | None = None
    min_speakers: int = 1
    max_speakers: int = 10

    # Paths
    vocabulary_path: str | None = None
    speaker_db_path: str | None = None

    # Resource management
    max_cached_models: int = Field(default=2, ge=1)
    num_workers: int = Field(default=1, ge=1)

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        """Add YAML source if available."""
        sources = list(kwargs.values())
        # Try to add YAML source
        try:
            from pydantic_settings import YamlConfigSettingsSource

            yaml_path = Path(cls._yaml_file)
            if yaml_path.is_file():
                yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=cls._yaml_file)
                sources.append(yaml_source)
        except ImportError:
            pass
        return tuple(sources)


def get_config() -> TranscriptyConfig:
    """Get the current configuration singleton.

    On first call, loads from YAML + env vars. If no config file exists,
    uses hardware-suggested defaults.
    """
    global _config
    if _config is not None:
        return _config

    with _config_lock:
        if _config is not None:
            return _config

        if _DEFAULT_CONFIG_FILE.is_file():
            logger.debug("Loading config from %s", _DEFAULT_CONFIG_FILE)
            _config = TranscriptyConfig()
        else:
            # Use hardware-suggested defaults if available
            try:
                from transcripty.hardware import detect_hardware

                profile = detect_hardware()
                suggestions = profile.suggest_settings()
                logger.info("No config file found, using hardware-suggested defaults")
                _config = TranscriptyConfig(**suggestions)
            except Exception:
                logger.debug("Hardware detection failed, using defaults")
                _config = TranscriptyConfig()

    return _config


def configure(**overrides: Any) -> TranscriptyConfig:
    """Update configuration with runtime overrides.

    Passing None explicitly (e.g. language=None) resets that field.
    To leave a field unchanged, omit it entirely.

    Args:
        **overrides: Config fields to override (e.g. model_size="medium").

    Returns:
        The updated configuration.
    """
    global _config
    with _config_lock:
        current = _config or TranscriptyConfig()
        merged = current.model_dump()
        merged.update(overrides)
        _config = TranscriptyConfig(**merged)
        logger.info("Configuration updated: %s", list(overrides.keys()))
    return _config


def reset_config() -> None:
    """Reset config singleton (for testing)."""
    global _config
    with _config_lock:
        _config = None
