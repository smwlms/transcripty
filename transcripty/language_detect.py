"""Post-processing language detection per segment using lingua-py.

Provides per-segment language detection to supplement faster-whisper's
global language detection. Uses a singleton LanguageDetector restricted
to NL/FR/EN for maximum accuracy in Belgian real estate context.

This module is optional: if lingua-language-detector is not installed,
all functions gracefully return None or pass through segments unchanged.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transcripty.models import Segment

logger = logging.getLogger(__name__)

# Singleton detector + lock
_detector = None
_detector_lock = threading.Lock()
_lingua_available: bool | None = None

# Minimum word count for reliable detection
MIN_WORDS_FOR_DETECTION = 3

# Lingua Language -> ISO 639-1 code mapping
_LINGUA_TO_ISO: dict[str, str] = {
    "DUTCH": "nl",
    "FRENCH": "fr",
    "ENGLISH": "en",
}


def _check_lingua_available() -> bool:
    """Check if lingua-language-detector is installed."""
    global _lingua_available
    if _lingua_available is not None:
        return _lingua_available
    try:
        import lingua  # noqa: F401

        _lingua_available = True
    except ImportError:
        _lingua_available = False
        logger.debug("lingua-language-detector not installed, language detection disabled")
    return _lingua_available


def _get_detector():
    """Get or create the singleton LanguageDetector (thread-safe).

    The detector is restricted to NL/FR/EN for accuracy.
    Lazy initialization: first call is slow (~1s), subsequent calls are instant.
    """
    global _detector

    if not _check_lingua_available():
        return None

    if _detector is not None:
        return _detector

    with _detector_lock:
        # Double-check after acquiring lock
        if _detector is not None:
            return _detector

        from lingua import Language, LanguageDetectorBuilder

        logger.info("Initializing lingua LanguageDetector (NL/FR/EN)...")
        _detector = (
            LanguageDetectorBuilder.from_languages(
                Language.DUTCH, Language.FRENCH, Language.ENGLISH
            )
            .with_preloaded_language_models()
            .build()
        )
        logger.info("Lingua LanguageDetector ready")
        return _detector


def detect_language(text: str) -> str | None:
    """Detect the language of a text string.

    Args:
        text: The text to detect the language of.

    Returns:
        ISO 639-1 language code ("nl", "fr", "en") or None if:
        - lingua is not installed
        - text has fewer than MIN_WORDS_FOR_DETECTION words
        - detection confidence is too low
    """
    if not text or len(text.split()) < MIN_WORDS_FOR_DETECTION:
        return None

    detector = _get_detector()
    if detector is None:
        return None

    result = detector.detect_language_of(text)
    if result is None:
        return None

    return _LINGUA_TO_ISO.get(result.name)


def detect_segment_languages(segments: list[Segment]) -> list[Segment]:
    """Detect and set the language for each segment using lingua-py.

    For segments with text shorter than MIN_WORDS_FOR_DETECTION words,
    the existing language value is preserved (too short for reliable detection).

    Args:
        segments: List of Segment objects from transcription.

    Returns:
        The same list of segments with updated language fields.
        If lingua is not installed, segments are returned unchanged.
    """
    if not _check_lingua_available():
        logger.debug("Skipping per-segment language detection (lingua not available)")
        return segments

    detector = _get_detector()
    if detector is None:
        return segments

    updated = 0
    for segment in segments:
        detected = detect_language(segment.text)
        if detected is not None:
            segment.language = detected
            updated += 1

    logger.info(
        "Per-segment language detection: %d/%d segments updated",
        updated,
        len(segments),
    )
    return segments


def reset_detector() -> None:
    """Reset the singleton detector (for testing)."""
    global _detector, _lingua_available
    with _detector_lock:
        _detector = None
        _lingua_available = None
