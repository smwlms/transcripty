"""Tests for output formatters."""

from transcripty.formatters import to_srt, to_text, to_vtt
from transcripty.models import LabeledSegment, Segment


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
