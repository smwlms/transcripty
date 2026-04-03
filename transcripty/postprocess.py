"""Post-processing corrections for Belgian real estate transcription.

Fixes common Whisper errors on Dutch/French proper nouns, real estate
jargon, and number formatting. Applied after transcription to reduce WER.

Usage:
    from transcripty.postprocess import correct_text, correct_segments
    fixed = correct_text("De EPC-lepel is B.")  # → "De EPC-label is B."
    correct_segments(segments)  # in-place correction
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Word-level corrections: whisper_output → correct
# Only include patterns that are consistently wrong across recordings
WORD_CORRECTIONS: dict[str, str] = {
    # Eigennamen Claes & Willems
    "klaas": "Claes",
    "klaes": "Claes",
    "class": "Claes",
    "willems": "Willems",
    # Vastgoed jargon
    "epc-lepel": "EPC-label",
    "epc lepel": "EPC-label",
    "epc label": "EPC-label",
    "rijbewoning": "rijwoning",
    "rijbehuizing": "rijwoning",
    # Homophones NL
    "bot": "bod",  # context: "een bot binnengekomen" → "een bod"
    "bots": "bods",
    # Samengestelde woorden die Whisper splitst
    "opschorten en voorwaarden": "opschortende voorwaarden",
    "opschorten de voorwaarden": "opschortende voorwaarden",
    "dienst baarheid": "erfdienstbaarheid",
    "dienstbaarheid": "erfdienstbaarheid",
    "vroeg probleem": "vochtprobleem",
    "vroegprobleem": "vochtprobleem",
    # Whisper NL eigenaardigheden
    "acten verleiden": "akte verlijden",
    "akte verleiden": "akte verlijden",
    "acte verlijden": "akte verlijden",
}

# Phrase-level corrections (multi-word patterns)
PHRASE_CORRECTIONS: list[tuple[re.Pattern, str]] = [
    # "van Klaas Willems" → "van Claes en Willems"
    (re.compile(r"\bvan\s+[Kk]la[ae]s\s+[Ww]illems\b"), "van Claes en Willems"),
    # "Claes Willems" without "en"
    (re.compile(r"\b[Cc]laes\s+[Ww]illems\b(?!\s+en)"), "Claes en Willems"),
    # "een bot binnengekomen" → "een bod binnengekomen"
    (re.compile(r"\been\s+bot\s+binnengekomen\b"), "een bod binnengekomen"),
    # Getallen: "driehonderdvijfentwintigduizend" patterns
    (re.compile(r"\b325\.000\s+euro\b", re.IGNORECASE), "325.000 euro"),
]


def correct_text(text: str) -> str:
    """Apply post-processing corrections to transcribed text.

    Fixes known Whisper errors on proper nouns, jargon, and homophones.
    Only applies high-confidence corrections that are consistently wrong.

    Args:
        text: Transcribed text to correct.

    Returns:
        Corrected text.
    """
    if not text:
        return text

    original = text
    corrected = text

    # Phrase corrections first (multi-word patterns)
    for pattern, replacement in PHRASE_CORRECTIONS:
        corrected = pattern.sub(replacement, corrected)

    # Word corrections (case-insensitive lookup)
    words = corrected.split()
    for i, word in enumerate(words):
        # Strip punctuation for lookup
        stripped = word.strip(".,!?;:\"'()[]")
        lower = stripped.lower()

        if lower in WORD_CORRECTIONS:
            replacement = WORD_CORRECTIONS[lower]
            # Preserve surrounding punctuation
            prefix = word[: word.index(stripped[0])] if stripped and stripped[0] in word else ""
            suffix = word[word.rindex(stripped[-1]) + 1 :] if stripped and stripped[-1] in word else ""
            words[i] = prefix + replacement + suffix

    corrected = " ".join(words)

    # Multi-word corrections that span word boundaries
    for wrong, right in WORD_CORRECTIONS.items():
        if " " in wrong:
            corrected = re.sub(
                re.escape(wrong), right, corrected, flags=re.IGNORECASE,
            )

    if corrected != original:
        n_changes = sum(1 for a, b in zip(original.split(), corrected.split()) if a != b)
        logger.debug("Post-processing: %d corrections applied", n_changes)

    return corrected


def correct_segments(segments: list) -> list:
    """Apply post-processing corrections to all segments in-place.

    Works with both Segment and LabeledSegment objects.

    Args:
        segments: List of segment objects with a .text attribute.

    Returns:
        The same list with corrected text fields.
    """
    corrections = 0
    for seg in segments:
        original = seg.text
        seg.text = correct_text(seg.text)
        if seg.text != original:
            corrections += 1

    if corrections:
        logger.info("Post-processing: %d/%d segments corrected", corrections, len(segments))

    return segments
