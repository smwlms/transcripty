"""Tests for diarize module (mocked — no actual pyannote model needed)."""

from unittest.mock import MagicMock, patch

from transcripty.config import reset_config
from transcripty.diarize import _pipeline_cache, _resolve_hf_token, reset_hf_token_cache


def test_resolve_hf_token_explicit():
    assert _resolve_hf_token("my-token") == "my-token"


def test_resolve_hf_token_env(monkeypatch):
    reset_hf_token_cache()
    reset_config()
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert _resolve_hf_token(None) == "env-token"
    reset_hf_token_cache()
    reset_config()


def test_resolve_hf_token_none_without_dotenv(monkeypatch):
    """Without dotenv and without env var, returns None."""
    reset_hf_token_cache()
    reset_config()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Patch the dotenv import to fail
    builtin_import = getattr(__builtins__, "__import__", __import__)
    original_import = builtin_import

    def mock_import(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("no dotenv")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        result = _resolve_hf_token(None)
    assert result is None
    reset_hf_token_cache()
    reset_config()


def test_pipeline_cache_key_format():
    """Pipeline cache key is pipeline_name:device."""
    _pipeline_cache.clear()

    sentinel = object()
    _pipeline_cache["pyannote/speaker-diarization-3.1:cpu"] = sentinel

    assert "pyannote/speaker-diarization-3.1:cpu" in _pipeline_cache
    assert "pyannote/speaker-diarization-3.1:mps" not in _pipeline_cache

    _pipeline_cache.clear()


def test_pipeline_cache_different_devices():
    """Different devices create different cache entries."""
    _pipeline_cache.clear()

    _pipeline_cache["pyannote/test:cpu"] = MagicMock()
    _pipeline_cache["pyannote/test:mps"] = MagicMock()

    assert len(_pipeline_cache) == 2
    assert _pipeline_cache["pyannote/test:cpu"] is not _pipeline_cache["pyannote/test:mps"]

    _pipeline_cache.clear()


@patch("transcripty.diarize._get_pipeline")
@patch("transcripty.diarize.wav_audio")
@patch("transcripty.diarize.detect_device", return_value="cpu")
@patch("transcripty.diarize._resolve_hf_token", return_value="token")
def test_diarize_basic(mock_token, mock_device, mock_wav, mock_get_pipeline, tmp_path):
    mock_torch = MagicMock()
    mock_pyannote = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "torch": mock_torch,
            "pyannote": MagicMock(),
            "pyannote.audio": mock_pyannote,
        },
    ):
        from transcripty.diarize import diarize

        audio = tmp_path / "test.wav"
        audio.touch()
        mock_wav.return_value.__enter__ = MagicMock(return_value=audio)
        mock_wav.return_value.__exit__ = MagicMock(return_value=False)

        mock_segment1 = MagicMock()
        mock_segment1.start = 0.0
        mock_segment1.end = 5.0

        mock_segment2 = MagicMock()
        mock_segment2.start = 5.0
        mock_segment2.end = 10.0

        # v3 style: pipeline returns annotation directly (no speaker_diarization attr)
        mock_annotation = MagicMock(spec=[])
        mock_annotation.itertracks = MagicMock(
            return_value=[
                (mock_segment1, None, "SPEAKER_00"),
                (mock_segment2, None, "SPEAKER_01"),
            ]
        )

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = mock_annotation
        mock_get_pipeline.return_value = mock_pipeline

        result = diarize(audio)

        assert result.num_speakers == 2
        assert len(result.segments) == 2
        assert result.segments[0].speaker == "SPEAKER_00"
        assert result.segments[1].speaker == "SPEAKER_01"


@patch("transcripty.diarize._get_pipeline")
@patch("transcripty.diarize.wav_audio")
@patch("transcripty.diarize.detect_device", return_value="cpu")
@patch("transcripty.diarize._resolve_hf_token", return_value="token")
def test_diarize_v4_output(mock_token, mock_device, mock_wav, mock_get_pipeline, tmp_path):
    """Test that pyannote v4 DiarizeOutput is handled correctly."""
    mock_torch = MagicMock()
    mock_pyannote = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "torch": mock_torch,
            "pyannote": MagicMock(),
            "pyannote.audio": mock_pyannote,
        },
    ):
        from transcripty.diarize import diarize

        audio = tmp_path / "test.wav"
        audio.touch()
        mock_wav.return_value.__enter__ = MagicMock(return_value=audio)
        mock_wav.return_value.__exit__ = MagicMock(return_value=False)

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 5.0

        mock_annotation = MagicMock()
        mock_annotation.itertracks.return_value = [
            (mock_segment, None, "SPEAKER_00"),
        ]

        # v4 style: output has speaker_diarization attribute
        mock_output = MagicMock()
        mock_output.speaker_diarization = mock_annotation
        mock_output.speaker_embeddings = None

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = mock_output
        mock_get_pipeline.return_value = mock_pipeline

        result = diarize(audio)

        assert result.num_speakers == 1
        assert result.segments[0].speaker == "SPEAKER_00"


@patch("transcripty.diarize._get_pipeline")
@patch("transcripty.diarize.wav_audio")
@patch("transcripty.diarize.detect_device", return_value="cpu")
@patch("transcripty.diarize._resolve_hf_token", return_value="token")
def test_diarize_with_num_speakers(mock_token, mock_device, mock_wav, mock_get_pipeline, tmp_path):
    """num_speakers parameter is passed to the pipeline."""
    mock_torch = MagicMock()

    modules = {
        "torch": mock_torch,
        "pyannote": MagicMock(),
        "pyannote.audio": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        from transcripty.diarize import diarize

        audio = tmp_path / "test.wav"
        audio.touch()
        mock_wav.return_value.__enter__ = MagicMock(return_value=audio)
        mock_wav.return_value.__exit__ = MagicMock(return_value=False)

        mock_annotation = MagicMock(spec=[])
        mock_annotation.itertracks = MagicMock(return_value=[])

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = mock_annotation
        mock_get_pipeline.return_value = mock_pipeline

        diarize(audio, num_speakers=2)

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["num_speakers"] == 2
        assert "min_speakers" not in call_kwargs
