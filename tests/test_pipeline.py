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
