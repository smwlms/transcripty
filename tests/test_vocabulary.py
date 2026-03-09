"""Tests for custom vocabulary."""

from pathlib import Path

from transcripty.vocabulary import Vocabulary


def test_create_vocabulary():
    vocab = Vocabulary(["Claes & Willems", "inkopen", "Whise"])
    assert len(vocab) == 3


def test_add_word():
    vocab = Vocabulary()
    vocab.add("Colibry")
    vocab.add("Colibry")  # duplicate
    assert len(vocab) == 1
    assert vocab.words == ["Colibry"]


def test_remove_word():
    vocab = Vocabulary(["a", "b", "c"])
    vocab.remove("b")
    assert vocab.words == ["a", "c"]


def test_as_prompt():
    vocab = Vocabulary(["Claes & Willems", "inkopen", "Whise"])
    assert vocab.as_prompt() == "Claes & Willems, inkopen, Whise"


def test_as_prompt_empty():
    vocab = Vocabulary()
    assert vocab.as_prompt() == ""


def test_save_and_load(tmp_path: Path):
    vocab = Vocabulary(["Claes & Willems", "inkopen", "Colibry"])
    path = tmp_path / "vocab.json"
    vocab.save(path)

    loaded = Vocabulary.load(path)
    assert loaded.words == vocab.words
    assert len(loaded) == 3


def test_repr():
    vocab = Vocabulary(["test"])
    assert "test" in repr(vocab)
