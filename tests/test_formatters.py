"""Tests for output formatters."""

from transcripty.formatters import to_srt, to_text, to_vtt, to_word_highlights
from transcripty.models import LabeledSegment, Segment, Word


def test_to_srt_basic():
    segments = [
        Segment(text="Hello world", start=0.0, end=2.5),
        Segment(text="Second line", start=3.0, end=5.0),
    ]
    result = to_srt(segments)
    assert "1\n00:00:00,000 --> 00:00:02,500\nHello world" in result
    assert "2\n00:00:03,000 --> 00:00:05,000\nSecond line" in result


def test_to_srt_with_speaker():
    segments = [
        LabeledSegment(text="Hello", start=0.0, end=1.0, speaker="Alice"),
    ]
    result = to_srt(segments)
    assert "[Alice] Hello" in result


def test_to_srt_long_timestamp():
    segments = [
        Segment(text="After one hour", start=3661.5, end=3665.0),
    ]
    result = to_srt(segments)
    assert "01:01:01,500" in result


def test_to_vtt_basic():
    segments = [
        Segment(text="Hello", start=0.0, end=1.0),
    ]
    result = to_vtt(segments)
    assert result.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.000" in result


def test_to_vtt_with_speaker():
    segments = [
        LabeledSegment(text="Hello", start=0.0, end=1.0, speaker="Bob"),
    ]
    result = to_vtt(segments)
    assert "<v Bob>Hello" in result


def test_to_text_basic():
    segments = [
        Segment(text="Hello world", start=0.0, end=1.0),
        Segment(text="Second line", start=1.5, end=3.0),
    ]
    result = to_text(segments)
    assert result == "Hello world\nSecond line"


def test_to_text_with_speakers():
    segments = [
        LabeledSegment(text="Hello", start=0.0, end=1.0, speaker="Alice"),
        LabeledSegment(text="Hi", start=1.0, end=2.0, speaker="Bob"),
    ]
    result = to_text(segments, include_speakers=True)
    assert "Alice: Hello" in result
    assert "Bob: Hi" in result


def test_to_text_with_timestamps():
    segments = [
        Segment(text="Hello", start=65.0, end=70.0),
    ]
    result = to_text(segments, include_timestamps=True)
    assert "[01:05]" in result


def test_to_text_no_speakers():
    segments = [
        LabeledSegment(text="Hello", start=0.0, end=1.0, speaker="Alice"),
    ]
    result = to_text(segments, include_speakers=False)
    assert "Alice" not in result
    assert result == "Hello"


def test_to_srt_empty():
    assert to_srt([]) == ""


def test_to_vtt_empty():
    result = to_vtt([])
    assert result.startswith("WEBVTT")


def test_to_text_empty():
    assert to_text([]) == ""


# --- Word highlights ---


def test_word_highlights_basic():
    segments = [
        Segment(
            text="Hello world",
            start=0.0,
            end=1.0,
            words=[
                Word(text="Hello", start=0.0, end=0.45, probability=0.95),
                Word(text="world", start=0.45, end=1.0, probability=0.92),
            ],
        ),
    ]
    highlights = to_word_highlights(segments)
    assert len(highlights) == 2
    assert highlights[0].word == "Hello"
    assert highlights[0].start == 0.0
    assert highlights[0].end == 0.45
    assert highlights[0].probability == 0.95
    assert highlights[0].segment_index == 0
    assert highlights[0].speaker is None
    assert highlights[1].word == "world"


def test_word_highlights_multiple_segments():
    segments = [
        Segment(
            text="First",
            start=0.0,
            end=1.0,
            words=[Word(text="First", start=0.0, end=1.0)],
        ),
        Segment(
            text="Second",
            start=1.5,
            end=2.5,
            words=[Word(text="Second", start=1.5, end=2.5)],
        ),
    ]
    highlights = to_word_highlights(segments)
    assert len(highlights) == 2
    assert highlights[0].segment_index == 0
    assert highlights[1].segment_index == 1


def test_word_highlights_with_speakers():
    segments = [
        LabeledSegment(
            text="Hi there",
            start=0.0,
            end=1.0,
            speaker="Alice",
            words=[
                Word(text="Hi", start=0.0, end=0.4),
                Word(text="there", start=0.4, end=1.0),
            ],
        ),
        LabeledSegment(
            text="Hello",
            start=1.5,
            end=2.0,
            speaker="Bob",
            words=[Word(text="Hello", start=1.5, end=2.0)],
        ),
    ]
    highlights = to_word_highlights(segments)
    assert len(highlights) == 3
    assert highlights[0].speaker == "Alice"
    assert highlights[1].speaker == "Alice"
    assert highlights[2].speaker == "Bob"


def test_word_highlights_empty():
    assert to_word_highlights([]) == []


def test_word_highlights_no_words():
    """Segments without word timestamps produce no highlights."""
    segments = [Segment(text="No words here", start=0.0, end=2.0)]
    assert to_word_highlights(segments) == []


def test_word_highlights_serialization():
    """WordHighlight serializes to the expected JSON structure."""
    segments = [
        Segment(
            text="Test",
            start=0.0,
            end=0.5,
            words=[Word(text="Test", start=0.0, end=0.5, probability=0.88)],
        ),
    ]
    highlights = to_word_highlights(segments)
    data = highlights[0].model_dump()
    assert data == {
        "word": "Test",
        "start": 0.0,
        "end": 0.5,
        "probability": 0.88,
        "segment_index": 0,
        "speaker": None,
    }
