"""Audio chunking for long-form processing.

Splits audio into time-based chunks via ffmpeg without loading the full
file into memory.  Each chunk is written to a temporary WAV file that
the caller is responsible for cleaning up (see ``cleanup_chunks``).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from transcripty.audio import audio_duration

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Metadata for a single audio chunk."""

    path: Path
    start: float  # start time in the original audio (seconds)
    end: float  # end time in the original audio (seconds)
    duration: float  # duration of this chunk (seconds)
    index: int  # 0-based chunk number


def split_audio(
    audio_path: str | Path,
    chunk_minutes: float = 10.0,
    overlap_seconds: float = 5.0,
) -> list[AudioChunk]:
    """Split audio into overlapping chunks using ffmpeg.

    Each chunk is written as a temporary mono 16-bit WAV file.  Chunks
    overlap by ``overlap_seconds`` so that segment boundaries falling at
    a split point can be deduplicated later.

    Args:
        audio_path: Path to the source audio file (any ffmpeg-supported format).
        chunk_minutes: Target length of each chunk in minutes.
        overlap_seconds: Overlap between consecutive chunks in seconds.

    Returns:
        Ordered list of :class:`AudioChunk` objects.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.
        RuntimeError: If ffmpeg is not installed.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is required for audio chunking. "
            "Install with: apt-get install ffmpeg (Linux) or brew install ffmpeg (macOS)"
        )

    total_duration = audio_duration(path)
    chunk_seconds = chunk_minutes * 60.0

    # Single chunk if audio is shorter than one chunk
    if total_duration <= chunk_seconds:
        return [_extract_chunk(ffmpeg, path, 0.0, total_duration, 0)]

    chunks: list[AudioChunk] = []
    start = 0.0
    index = 0

    while start < total_duration:
        end = min(start + chunk_seconds, total_duration)
        chunk = _extract_chunk(ffmpeg, path, start, end - start, index)
        chunks.append(chunk)
        index += 1
        # Advance by chunk length minus overlap
        start += chunk_seconds - overlap_seconds
        if start >= total_duration:
            break

    logger.info(
        "Split %s into %d chunks (%.1f min each, %.1fs overlap)",
        path.name,
        len(chunks),
        chunk_minutes,
        overlap_seconds,
    )
    return chunks


def _extract_chunk(
    ffmpeg: str,
    source: Path,
    start: float,
    duration: float,
    index: int,
) -> AudioChunk:
    """Extract a single chunk from the source audio via ffmpeg."""
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix=f"chunk_{index:03d}_")
    os.close(fd)

    result = subprocess.run(
        [
            ffmpeg,
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            str(source),
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-y",
            tmp_path,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        # Clean up on failure
        Path(tmp_path).unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg failed extracting chunk {index} "
            f"(start={start:.1f}s, duration={duration:.1f}s): "
            f"{result.stderr[:200]}"
        )

    return AudioChunk(
        path=Path(tmp_path),
        start=start,
        end=start + duration,
        duration=duration,
        index=index,
    )


def cleanup_chunks(chunks: list[AudioChunk]) -> None:
    """Remove temporary chunk files from disk."""
    for chunk in chunks:
        if chunk.path.exists():
            chunk.path.unlink()
            logger.debug("Cleaned up chunk %d: %s", chunk.index, chunk.path.name)
