"""Code-switching detection within segments.

Detects language switches within a single transcription segment,
common in Belgian real estate conversations (NL/FR mixed).

Uses lingua's detect_multiple_languages_of() for reliable multi-language
detection within a single text span.

Usage:
    from transcripty.codeswitching import detect_code_switches, analyze_segment
    switches = detect_code_switches("de notaris zegt que c'est pas possible")
    # [CodeSwitch(position=17, from_lang="nl", to_lang="fr", ...)]

    profile = analyze_segment("c'est une belle maison met een grote tuin")
    # SegmentLanguageProfile(spans=[("fr", "c'est une belle maison"), ("nl", "met een grote tuin")])
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_detector = None
_detector_lock = threading.Lock()

_LINGUA_TO_ISO = {"DUTCH": "nl", "FRENCH": "fr", "ENGLISH": "en"}


def _get_detector():
    """Get or create lingua detector (singleton)."""
    global _detector
    if _detector is not None:
        return _detector

    with _detector_lock:
        if _detector is not None:
            return _detector
        try:
            from lingua import Language, LanguageDetectorBuilder

            _detector = (
                LanguageDetectorBuilder.from_languages(
                    Language.DUTCH, Language.FRENCH, Language.ENGLISH,
                )
                .with_preloaded_language_models()
                .build()
            )
        except ImportError:
            logger.debug("lingua not installed, code-switching detection disabled")
            return None
    return _detector


@dataclass
class LanguageSpan:
    """A contiguous span of text in one language."""

    language: str  # ISO code: nl, fr, en
    text: str
    start_char: int  # character offset in original text
    end_char: int


@dataclass
class CodeSwitch:
    """A detected language switch point within a segment."""

    char_position: int  # character index where the switch occurs
    from_lang: str
    to_lang: str
    context_before: str  # last ~30 chars before switch
    context_after: str  # first ~30 chars after switch


@dataclass
class SegmentLanguageProfile:
    """Language profile for a segment with possible code-switches."""

    dominant_language: str
    is_multilingual: bool
    spans: list[LanguageSpan] = field(default_factory=list)
    switches: list[CodeSwitch] = field(default_factory=list)


def detect_code_switches(text: str) -> list[CodeSwitch]:
    """Detect language switch points in a text.

    Uses lingua's detect_multiple_languages_of() for reliable detection.
    Only reports actual switches (ignores single-language texts).

    Args:
        text: The text to analyze.

    Returns:
        List of CodeSwitch objects at each detected switch point.
    """
    detector = _get_detector()
    if detector is None or not text or len(text.split()) < 3:
        return []

    try:
        results = detector.detect_multiple_languages_of(text)
    except Exception:
        return []

    if len(results) <= 1:
        return []

    switches = []
    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]

        from_lang = _LINGUA_TO_ISO.get(prev.language.name, "?")
        to_lang = _LINGUA_TO_ISO.get(curr.language.name, "?")

        if from_lang == to_lang:
            continue

        # Context around the switch point
        switch_pos = curr.start_index
        context_before = text[max(0, switch_pos - 30) : switch_pos].strip()
        context_after = text[switch_pos : switch_pos + 30].strip()

        switches.append(CodeSwitch(
            char_position=switch_pos,
            from_lang=from_lang,
            to_lang=to_lang,
            context_before=context_before,
            context_after=context_after,
        ))

    return switches


def analyze_segment(text: str) -> SegmentLanguageProfile:
    """Full language analysis of a segment.

    Returns the dominant language, whether it's multilingual,
    language spans, and switch points.

    Args:
        text: Segment text to analyze.

    Returns:
        SegmentLanguageProfile with full analysis.
    """
    detector = _get_detector()
    if detector is None or not text:
        return SegmentLanguageProfile(dominant_language="?", is_multilingual=False)

    try:
        results = detector.detect_multiple_languages_of(text)
    except Exception:
        # Fallback to single-language detection
        result = detector.detect_language_of(text)
        lang = _LINGUA_TO_ISO.get(result.name, "?") if result else "?"
        return SegmentLanguageProfile(dominant_language=lang, is_multilingual=False)

    if not results:
        return SegmentLanguageProfile(dominant_language="?", is_multilingual=False)

    # Build spans
    spans = []
    lang_char_counts: dict[str, int] = {}

    for r in results:
        lang = _LINGUA_TO_ISO.get(r.language.name, "?")
        span_text = text[r.start_index : r.end_index]
        spans.append(LanguageSpan(
            language=lang,
            text=span_text,
            start_char=r.start_index,
            end_char=r.end_index,
        ))
        lang_char_counts[lang] = lang_char_counts.get(lang, 0) + len(span_text)

    # Dominant language = most characters
    dominant = max(lang_char_counts, key=lang_char_counts.get) if lang_char_counts else "?"
    is_multilingual = len(set(s.language for s in spans)) > 1

    # Detect switches
    switches = detect_code_switches(text)

    return SegmentLanguageProfile(
        dominant_language=dominant,
        is_multilingual=is_multilingual,
        spans=spans,
        switches=switches,
    )


def tag_segments_with_switches(segments: list) -> list:
    """Analyze all segments for code-switching.

    Adds `_code_switches` and `_language_spans` attributes to segments
    that contain language switches.

    Args:
        segments: List of segment objects with .text attribute.

    Returns:
        Same list with code-switching metadata added.
    """
    n_multilingual = 0
    n_switches = 0

    for seg in segments:
        profile = analyze_segment(seg.text)

        if profile.is_multilingual:
            n_multilingual += 1
            n_switches += len(profile.switches)

            seg._language_spans = [
                {"lang": s.language, "text": s.text}
                for s in profile.spans
            ]
            seg._code_switches = [
                {
                    "from": s.from_lang,
                    "to": s.to_lang,
                    "position": s.char_position,
                    "before": s.context_before,
                    "after": s.context_after,
                }
                for s in profile.switches
            ]

    if n_multilingual:
        logger.info(
            "Code-switching: %d multilingual segments, %d switches",
            n_multilingual, n_switches,
        )

    return segments
