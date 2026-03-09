"""Tests for merge logic."""

from transcripty.merge import merge
from transcripty.models import DiarizationSegment, Segment


def test_merge_basic():
    segments = [
        Segment(text="Hello there", start=0.0, end=2.0),
        Segment(text="How are you", start=2.5, end=4.0),
    ]
    diarization = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.5),
        DiarizationSegment(speaker="SPEAKER_01", start=2.5, end=5.0),
    ]

    result = merge(segments, diarization)

    assert len(result) == 2
    assert result[0].speaker == "SPEAKER_00"
    assert result[0].text == "Hello there"
    assert result[1].speaker == "SPEAKER_01"
    assert result[1].text == "How are you"


def test_merge_overlap_picks_dominant_speaker():
    """When a segment overlaps two speakers, pick the one with more overlap."""
    segments = [
        Segment(text="Overlapping", start=1.0, end=4.0),
    ]
    diarization = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),  # 1s overlap
        DiarizationSegment(speaker="SPEAKER_01", start=2.0, end=5.0),  # 2s overlap
    ]

    result = merge(segments, diarization)
    assert result[0].speaker == "SPEAKER_01"


def test_merge_no_diarization_match():
    """Segments with no diarization overlap get UNKNOWN speaker."""
    segments = [
        Segment(text="Silence gap", start=10.0, end=12.0),
    ]
    diarization = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=5.0),
    ]

    result = merge(segments, diarization)
    assert result[0].speaker == "UNKNOWN"


def test_merge_empty_segments():
    result = merge([], [])
    assert result == []


def test_merge_empty_diarization():
    segments = [Segment(text="Hello", start=0.0, end=1.0)]
    result = merge(segments, [])
    assert len(result) == 1
    assert result[0].speaker == "UNKNOWN"


def test_merge_preserves_words():
    from transcripty.models import Word

    words = [Word(text="Hello", start=0.0, end=0.5, probability=0.9)]
    segments = [Segment(text="Hello", start=0.0, end=1.0, words=words)]
    diarization = [DiarizationSegment(speaker="SPK", start=0.0, end=1.0)]

    result = merge(segments, diarization)
    assert len(result[0].words) == 1
    assert result[0].words[0].text == "Hello"


def test_merge_multiple_diarization_segments_same_speaker():
    """Speaker with fragmented diarization segments should accumulate overlap."""
    segments = [
        Segment(text="Long sentence", start=0.0, end=10.0),
    ]
    diarization = [
        DiarizationSegment(speaker="A", start=0.0, end=3.0),  # 3s
        DiarizationSegment(speaker="B", start=3.0, end=5.0),  # 2s
        DiarizationSegment(speaker="A", start=5.0, end=10.0),  # 5s -> A total = 8s
    ]

    result = merge(segments, diarization)
    assert result[0].speaker == "A"
