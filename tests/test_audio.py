"""Tests for audio conversion utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcripty.audio import wav_audio


def test_wav_passthrough(tmp_path: Path):
    """WAV files should be yielded directly without conversion."""
    wav_file = tmp_path / "test.wav"
    wav_file.write_bytes(b"RIFF" + b"\x00" * 100)

    with wav_audio(wav_file) as result:
        assert result == wav_file


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        with wav_audio("/nonexistent/file.mp3"):
            pass


def test_non_wav_requires_pydub(tmp_path: Path):
    """Non-WAV files should raise ImportError if pydub is missing."""
    mp3_file = tmp_path / "test.mp3"
    mp3_file.write_bytes(b"\x00" * 100)

    with patch.dict("sys.modules", {"pydub": None}):
        with pytest.raises(ImportError, match="pydub"):
            with wav_audio(mp3_file):
                pass


def test_non_wav_converts_and_cleans_up(tmp_path: Path):
    """Non-WAV files should be converted and temp file cleaned up."""
    mp3_file = tmp_path / "test.mp3"
    mp3_file.write_bytes(b"\x00" * 100)

    mock_audio = MagicMock()
    mock_audio_segment = MagicMock()
    mock_audio_segment.from_file.return_value = mock_audio

    with patch("transcripty.audio.AudioSegment", mock_audio_segment, create=True):
        # We need to patch the import inside the function
        mock_pydub = MagicMock()
        mock_pydub.AudioSegment = mock_audio_segment
        with patch.dict("sys.modules", {"pydub": mock_pydub}):

            def fake_export(path, format):
                Path(path).write_bytes(b"RIFF" + b"\x00" * 50)

            mock_audio.export.side_effect = fake_export

            with wav_audio(mp3_file) as result:
                assert result.suffix == ".wav"
                temp_path = result  # Save for later check

            # Temp file should be cleaned up
            assert not temp_path.exists()
