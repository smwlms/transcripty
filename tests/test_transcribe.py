"""Tests for transcribe module (mocked — no actual Whisper model needed)."""

from unittest.mock import MagicMock, patch

from transcripty.transcribe import _model_cache, transcribe


def _make_mock_segment(text="Hello", start=0.0, end=1.0, words=None):
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    seg.words = words or []
    return seg


def _make_mock_info(language="en", language_probability=0.95, duration=10.0):
    info = MagicMock()
    info.language = language
    info.language_probability = language_probability
    info.duration = duration
    return info


@patch("transcripty.transcribe.wav_audio")
@patch("transcripty.transcribe.detect_device", return_value="cpu")
def test_transcribe_basic(mock_device, mock_wav, tmp_path):
    _model_cache.clear()

    audio = tmp_path / "test.wav"
    audio.touch()
    mock_wav.return_value.__enter__ = MagicMock(return_value=audio)
    mock_wav.return_value.__exit__ = MagicMock(return_value=False)

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [_make_mock_segment("Hello world", 0.0, 2.0)],
        _make_mock_info(),
    )

    with patch("transcripty.transcribe._get_model", return_value=mock_model):
        result = transcribe(audio)

    assert len(result.segments) == 1
    assert result.segments[0].text == "Hello world"
    assert result.language == "en"
    assert result.duration == 10.0


@patch("transcripty.transcribe.wav_audio")
@patch("transcripty.transcribe.detect_device", return_value="cpu")
def test_transcribe_with_prompt(mock_device, mock_wav, tmp_path):
    _model_cache.clear()

    audio = tmp_path / "test.wav"
    audio.touch()
    mock_wav.return_value.__enter__ = MagicMock(return_value=audio)
    mock_wav.return_value.__exit__ = MagicMock(return_value=False)

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [_make_mock_segment()],
        _make_mock_info(),
    )

    with patch("transcripty.transcribe._get_model", return_value=mock_model):
        transcribe(audio, prompt="TensorFlow, Kubernetes, API")

    call_kwargs = mock_model.transcribe.call_args[1]
    assert call_kwargs["initial_prompt"] == "TensorFlow, Kubernetes, API"


@patch("transcripty.transcribe.wav_audio")
@patch("transcripty.transcribe.detect_device", return_value="cpu")
def test_transcribe_word_timestamps(mock_device, mock_wav, tmp_path):
    _model_cache.clear()

    audio = tmp_path / "test.wav"
    audio.touch()
    mock_wav.return_value.__enter__ = MagicMock(return_value=audio)
    mock_wav.return_value.__exit__ = MagicMock(return_value=False)

    mock_word = MagicMock()
    mock_word.word = "Hello"
    mock_word.start = 0.0
    mock_word.end = 0.5
    mock_word.probability = 0.95

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [_make_mock_segment("Hello", 0.0, 1.0, words=[mock_word])],
        _make_mock_info(),
    )

    with patch("transcripty.transcribe._get_model", return_value=mock_model):
        result = transcribe(audio, word_timestamps=True)

    assert len(result.segments[0].words) == 1
    assert result.segments[0].words[0].text == "Hello"
    assert result.segments[0].words[0].probability == 0.95


def test_model_caching():
    """Model cache stores by key and returns same instance for same key."""
    _model_cache.clear()

    sentinel_a = MagicMock(name="model_a")
    sentinel_b = MagicMock(name="model_b")

    _model_cache.get_or_load("small:int8:cpu", lambda: sentinel_a)
    _model_cache.get_or_load("medium:int8:cpu", lambda: sentinel_b)

    result_a = _model_cache.get_or_load("small:int8:cpu", lambda: MagicMock())
    result_b = _model_cache.get_or_load("medium:int8:cpu", lambda: MagicMock())
    assert result_a is sentinel_a
    assert result_b is sentinel_b
    assert len(_model_cache) == 2

    _model_cache.clear()


def test_model_cache_key_format():
    """Cache key format is model_size:compute_type:device."""
    _model_cache.clear()

    sentinel = object()
    _model_cache.get_or_load("large-v3:float16:cuda", lambda: sentinel)

    assert "large-v3:float16:cuda" in _model_cache
    assert "large-v3:int8:cuda" not in _model_cache

    _model_cache.clear()
