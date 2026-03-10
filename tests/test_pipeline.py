"""Tests for convenience pipeline."""

import sys
from unittest.mock import MagicMock, patch

import transcripty.pipeline
from transcripty.models import DiarizationResult, DiarizationSegment, Segment, TranscriptionResult
from transcripty.pipeline import transcribe_with_speakers


def test_transcribe_with_speakers(tmp_path):
    audio = tmp_path / "test.wav"
    audio.touch()

    mock_transcribe_result = TranscriptionResult(
        segments=[
            Segment(text="Hello", start=0.0, end=2.0),
            Segment(text="Hi there", start=2.5, end=4.0),
        ],
        language="en",
        language_probability=0.95,
        duration=4.0,
    )

    mock_diarize_result = DiarizationResult(
        segments=[
            DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.5),
            DiarizationSegment(speaker="SPEAKER_01", start=2.5, end=5.0),
        ],
        num_speakers=2,
    )

    mock_diarize_fn = MagicMock(return_value=mock_diarize_result)
    mock_diarize_module = MagicMock()
    mock_diarize_module.diarize = mock_diarize_fn

    with (
        patch.object(transcripty.pipeline, "transcribe", return_value=mock_transcribe_result),
        patch.dict(sys.modules, {"transcripty.diarize": mock_diarize_module}),
    ):
        result = transcribe_with_speakers(audio)

    assert len(result) == 2
    assert result[0].speaker == "SPEAKER_00"
    assert result[0].text == "Hello"
    assert result[1].speaker == "SPEAKER_01"


def test_transcribe_with_speakers_and_db(tmp_path):
    audio = tmp_path / "test.wav"
    audio.touch()

    mock_transcribe_result = TranscriptionResult(
        segments=[Segment(text="Hello", start=0.0, end=2.0)],
        language="en",
        language_probability=0.95,
        duration=2.0,
    )

    mock_diarize_result = DiarizationResult(
        segments=[DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0)],
        num_speakers=1,
        embeddings={"SPEAKER_00": [0.1, 0.2, 0.3]},
    )

    mock_db = MagicMock()
    mock_db.identify.return_value = {"SPEAKER_00": "Alice"}

    mock_diarize_fn = MagicMock(return_value=mock_diarize_result)
    mock_diarize_module = MagicMock()
    mock_diarize_module.diarize = mock_diarize_fn

    with (
        patch.object(transcripty.pipeline, "transcribe", return_value=mock_transcribe_result),
        patch.dict(sys.modules, {"transcripty.diarize": mock_diarize_module}),
    ):
        result = transcribe_with_speakers(audio, speaker_db=mock_db)

    assert result[0].speaker == "Alice"
    mock_db.identify.assert_called_once()


def test_transcribe_with_speakers_on_progress(tmp_path):
    """on_progress callback receives weighted progress across all stages."""
    audio = tmp_path / "test.wav"
    audio.touch()

    mock_transcribe_result = TranscriptionResult(
        segments=[Segment(text="Hello", start=0.0, end=2.0)],
        language="en",
        language_probability=0.95,
        duration=2.0,
    )

    mock_diarize_result = DiarizationResult(
        segments=[DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0)],
        num_speakers=1,
    )

    def mock_transcribe(path, on_progress=None, **kwargs):
        if on_progress:
            on_progress(0.0, "Transcribing...")
            on_progress(1.0, "Transcription complete")
        return mock_transcribe_result

    def mock_diarize(path, hf_token=None, num_speakers=None, on_progress=None):
        if on_progress:
            on_progress(0.0, "Diarizing...")
            on_progress(1.0, "Diarization complete")
        return mock_diarize_result

    mock_diarize_module = MagicMock()
    mock_diarize_module.diarize = mock_diarize

    progress_calls = []

    def on_progress(progress, message):
        progress_calls.append((round(progress, 2), message))

    with (
        patch.object(transcripty.pipeline, "transcribe", side_effect=mock_transcribe),
        patch.dict(sys.modules, {"transcripty.diarize": mock_diarize_module}),
    ):
        result = transcribe_with_speakers(audio, on_progress=on_progress)

    assert len(result) == 1
    # Verify weighted progress: transcribe 0-0.7, diarize 0.7-0.9, merge 0.9-1.0
    assert progress_calls[0] == (0.0, "Transcribing...")
    assert progress_calls[1] == (0.7, "Transcription complete")
    assert progress_calls[2] == (0.7, "Diarizing...")
    assert progress_calls[3] == (0.9, "Diarization complete")
    assert progress_calls[4] == (0.9, "Merging segments...")
    assert progress_calls[5] == (1.0, "Complete")
