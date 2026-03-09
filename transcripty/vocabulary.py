"""Custom vocabulary for improving Whisper recognition of domain-specific words."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Vocabulary:
    """A list of domain-specific words/phrases to improve transcription accuracy.

    Whisper uses an `initial_prompt` to bias recognition toward specific terms.
    This class manages a word list and generates the prompt string.

    Usage:
        vocab = Vocabulary(["Claes & Willems", "inkopen", "Whise", "Colibry"])
        result = transcribe("audio.mp3", prompt=vocab.as_prompt())
    """

    def __init__(self, words: list[str] | None = None) -> None:
        self.words: list[str] = words or []

    def add(self, word: str) -> None:
        """Add a word to the vocabulary (no duplicates)."""
        if word not in self.words:
            self.words.append(word)

    def remove(self, word: str) -> None:
        """Remove a word from the vocabulary."""
        self.words = [w for w in self.words if w != word]

    def as_prompt(self) -> str:
        """Generate a Whisper initial_prompt string from the word list."""
        return ", ".join(self.words)

    def save(self, path: str | Path) -> None:
        """Save vocabulary to a JSON file."""
        path = Path(path)
        data = {"words": self.words}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Vocabulary saved to %s (%d words)", path, len(self.words))

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        """Load vocabulary from a JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        words = data.get("words", [])
        logger.info("Vocabulary loaded from %s (%d words)", path, len(words))
        return cls(words=words)

    def __len__(self) -> int:
        return len(self.words)

    def __repr__(self) -> str:
        return f"Vocabulary({self.words!r})"
