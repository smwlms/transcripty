"""Audio conversion utilities.

Uses ffmpeg directly for memory-efficient conversion (no full-file RAM load).
Falls back to pydub when ffmpeg is not available.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

_ffmpeg_bin: str | None = None


def _find_ffmpeg() -> str | None:
    """Find ffmpeg binary, cached after first lookup."""
    global _ffmpeg_bin
    if _ffmpeg_bin is None:
        _ffmpeg_bin = shutil.which("ffmpeg") or ""
    return _ffmpeg_bin or None


def _find_ffprobe() -> str | None:
    """Find ffprobe binary."""
    return shutil.which("ffprobe")


def audio_duration(audio_path: str | Path) -> float:
    """Get audio duration in seconds without loading the file into memory.

    Uses ffprobe when available, falls back to pydub.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in seconds.

    Raises:
        FileNotFoundError: If audio_path does not exist.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ffprobe = _find_ffprobe()
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError):
            logger.debug("ffprobe duration detection failed, falling back to pydub")

    # Fallback to pydub (loads file into memory)
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(path))
    return len(audio) / 1000.0


def _convert_with_ffmpeg(input_path: Path, output_path: Path) -> bool:
    """Convert audio to WAV using ffmpeg subprocess (memory-efficient).

    Returns True on success, False if ffmpeg is unavailable or fails.
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return False

    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-i",
                str(input_path),
                "-f",
                "wav",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",  # mono
                "-y",  # overwrite
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max for very long files
        )
        if result.returncode == 0:
            logger.info("Converted to WAV via ffmpeg: %s", output_path.name)
            return True
        logger.warning("ffmpeg conversion failed: %s", result.stderr[:200])
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg conversion timed out for %s", input_path.name)

    return False


@contextmanager
def wav_audio(audio_path: str | Path) -> Generator[Path, None, None]:
    """Context manager that yields a WAV file path.

    If the input is already WAV, yields it directly.
    Otherwise converts to a temporary WAV file using ffmpeg (memory-efficient)
    or pydub as fallback, and cleans up after.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() == ".wav":
        logger.debug("Input is already WAV: %s", path.name)
        yield path
        return

    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    temp = Path(temp_path)

    try:
        # Try ffmpeg first (no RAM load)
        if _convert_with_ffmpeg(path, temp):
            yield temp
            return

        # Fallback to pydub (loads file into memory)
        logger.info("Falling back to pydub for %s conversion...", path.name)
        try:
            from pydub import AudioSegment
        except ImportError as e:
            raise ImportError(
                "pydub is required for audio conversion when ffmpeg is not "
                "available. Install with: pip install pydub"
            ) from e

        audio = AudioSegment.from_file(str(path))
        audio.export(str(temp), format="wav")
        logger.info("Converted to WAV via pydub: %s", temp.name)
        yield temp
    finally:
        if temp.exists():
            temp.unlink()
            logger.debug("Cleaned up temporary WAV file.")
