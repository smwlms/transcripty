"""Tests for per-segment language detection using lingua-py."""

import pytest

from transcripty.language_detect import (
    detect_language,
    detect_segment_languages,
    reset_detector,
)
from transcripty.models import Segment


@pytest.fixture(autouse=True)
def _reset_lingua():
    """Reset the lingua detector singleton between tests."""
    reset_detector()
    yield
    reset_detector()


class TestDetectLanguage:
    """Tests for detect_language() on standalone text."""

    def test_detects_dutch(self):
        text = "Dit is een test in het Nederlands om de taaldetectie te controleren"
        result = detect_language(text)
        assert result == "nl"

    def test_detects_french(self):
        text = "Ceci est un test en français pour vérifier la détection de langue"
        result = detect_language(text)
        assert result == "fr"

    def test_detects_english(self):
        text = "This is a test in English to verify the language detection"
        result = detect_language(text)
        assert result == "en"

    def test_short_text_returns_none(self):
        """Text with fewer than MIN_WORDS_FOR_DETECTION words should return None."""
        text = "Ja nee"  # 2 words, below threshold
        result = detect_language(text)
        assert result is None

    def test_empty_text_returns_none(self):
        result = detect_language("")
        assert result is None

    def test_exactly_min_words(self):
        """Text with exactly MIN_WORDS_FOR_DETECTION words should be detected."""
        # 3 words, at the threshold
        text = "Dit is goed"
        result = detect_language(text)
        # Should return a language (likely nl), not None
        assert result is not None

    def test_none_text_returns_none(self):
        """Passing None-like empty string returns None."""
        result = detect_language("")
        assert result is None


class TestDetectSegmentLanguages:
    """Tests for detect_segment_languages() on segment lists."""

    def test_updates_segment_languages(self):
        segments = [
            Segment(text="Dit is een Nederlands segment met genoeg woorden", start=0.0, end=2.0),
            Segment(text="This is an English segment with enough words", start=2.0, end=4.0),
            Segment(
                text="Ceci est un segment en français avec assez de mots",
                start=4.0,
                end=6.0,
            ),
        ]

        result = detect_segment_languages(segments)

        assert result[0].language == "nl"
        assert result[1].language == "en"
        assert result[2].language == "fr"

    def test_preserves_short_segment_language(self):
        """Short segments should keep their original language value."""
        segments = [
            Segment(text="Ja", start=0.0, end=0.5, language="nl"),
            Segment(
                text="Dit is een langer segment met genoeg woorden voor detectie",
                start=0.5,
                end=3.0,
                language="en",  # wrong language, should be overwritten
            ),
        ]

        result = detect_segment_languages(segments)

        # Short segment keeps original language
        assert result[0].language == "nl"
        # Long segment gets updated to correct language
        assert result[1].language == "nl"

    def test_returns_same_list(self):
        """Should modify in-place and return the same list object."""
        segments = [
            Segment(text="Dit is een test segment voor de taaldetectie", start=0.0, end=2.0),
        ]

        result = detect_segment_languages(segments)

        assert result is segments

    def test_empty_segments(self):
        """Empty segment list should return empty list."""
        result = detect_segment_languages([])
        assert result == []

    def test_mixed_language_conversation(self):
        """Simulate a bilingual conversation typical in Belgian real estate."""
        segments = [
            Segment(
                text="Goedemiddag, ik zou graag dit appartement willen bezichtigen",
                start=0.0,
                end=3.0,
                language="nl",  # from whisper global detection
            ),
            Segment(
                text="Bien sûr, je vous montre le salon et la cuisine",
                start=3.0,
                end=6.0,
                language="nl",  # whisper wrongly assigned global language
            ),
            Segment(
                text="En wat is de prijs per maand voor de huur",
                start=6.0,
                end=9.0,
                language="nl",
            ),
        ]

        result = detect_segment_languages(segments)

        assert result[0].language == "nl"
        assert result[1].language == "fr"  # corrected by lingua
        assert result[2].language == "nl"
