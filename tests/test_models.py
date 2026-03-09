"""Tests for Pydantic models."""

from transcripty.models import (
    DiarizationResult,
    DiarizationSegment,
    LabeledSegment,
    Segment,
    TranscriptionResult,
    Word,
)


def test_word_model():
    w = Word(text="hello", start=0.0, end=0.5, probability=0.95)
    assert w.text == "hello"
    assert w.probability == 0.95


def test_word_default_probability():
    w = Word(text="hi", start=0.0, end=0.1)
    assert w.probability == 0.0


def test_segment_model():
    seg = Segment(text="Hello world", start=0.0, end=1.5)
    assert seg.text == "Hello world"
    assert seg.words == []


def test_segment_with_words():
    words = [
        Word(text="Hello", start=0.0, end=0.5, probability=0.9),
        Word(text="world", start=0.6, end=1.0, probability=0.85),
    ]
    seg = Segment(text="Hello world", start=0.0, end=1.0, words=words)
    assert len(seg.words) == 2


def test_transcription_result():
    result = TranscriptionResult(
        segments=[Segment(text="Test", start=0.0, end=1.0)],
        language="en",
        language_probability=0.98,
        duration=10.5,
    )
    assert result.language == "en"
    assert len(result.segments) == 1
    assert result.duration == 10.5


def test_diarization_segment():
    ds = DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=5.0)
    assert ds.speaker == "SPEAKER_00"


def test_diarization_result():
    dr = DiarizationResult(
        segments=[
            DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=5.0),
            DiarizationSegment(speaker="SPEAKER_01", start=5.0, end=10.0),
        ],
        num_speakers=2,
    )
    assert dr.num_speakers == 2
    assert len(dr.segments) == 2


def test_labeled_segment():
    ls = LabeledSegment(
        text="Hello",
        start=0.0,
        end=1.0,
        speaker="SPEAKER_00",
    )
    assert ls.speaker == "SPEAKER_00"
    assert ls.words == []


def test_model_serialization():
    result = TranscriptionResult(
        segments=[Segment(text="Test", start=0.0, end=1.0)],
        language="nl",
        language_probability=0.95,
        duration=5.0,
    )
    data = result.model_dump()
    assert data["language"] == "nl"
    assert len(data["segments"]) == 1

    # Round-trip
    restored = TranscriptionResult.model_validate(data)
    assert restored == result
