"""Audio conversion utilities."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def wav_audio(audio_path: str | Path) -> Generator[Path, None, None]:
    """Context manager that yields a WAV file path.

    If the input is already WAV, yields it directly.
    Otherwise converts to a temporary WAV file using pydub and cleans up after.

    Requires pydub and ffmpeg for non-WAV inputs.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() == ".wav":
        logger.debug("Input is already WAV: %s", path.name)
        yield path
        return

    # Convert to temporary WAV
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise ImportError(
            "pydub is required for non-WAV audio conversion. Install with: pip install pydub"
        ) from e

    logger.info("Converting %s to WAV...", path.name)
    audio = AudioSegment.from_file(str(path))

    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    temp = Path(temp_path)

    try:
        audio.export(str(temp), format="wav")
        logger.info("Converted to temporary WAV: %s", temp.name)
        yield temp
    finally:
        if temp.exists():
            temp.unlink()
            logger.debug("Cleaned up temporary WAV file.")
