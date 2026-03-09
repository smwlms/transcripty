"""Tests for speaker enrollment and identification."""

from pathlib import Path

import pytest

from transcripty.models import DiarizationResult, DiarizationSegment
from transcripty.speakers import SpeakerDB, _cosine_similarity


def test_cosine_similarity_identical():
    a = [1.0, 0.0, 0.0]
    assert _cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert _cosine_similarity(a, b) == pytest.approx(-1.0)


def test_enroll_from_embedding():
    db = SpeakerDB()
    db.enroll_from_embedding("Alice", [0.1, 0.2, 0.3])
    assert "Alice" in db.names
    assert len(db) == 1


def test_identify_match():
    db = SpeakerDB()
    db.enroll_from_embedding("Alice", [1.0, 0.0, 0.0])
    db.enroll_from_embedding("Bob", [0.0, 1.0, 0.0])

    result = DiarizationResult(
        segments=[DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=5.0)],
        num_speakers=1,
        embeddings={"SPEAKER_00": [0.95, 0.05, 0.0]},  # close to Alice
    )

    names = db.identify(result, threshold=0.5)
    assert names == {"SPEAKER_00": "Alice"}


def test_identify_no_match_below_threshold():
    db = SpeakerDB()
    db.enroll_from_embedding("Alice", [1.0, 0.0, 0.0])

    result = DiarizationResult(
        segments=[DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=5.0)],
        num_speakers=1,
        embeddings={"SPEAKER_00": [0.0, 1.0, 0.0]},  # orthogonal to Alice
    )

    names = db.identify(result, threshold=0.5)
    assert names == {}


def test_identify_multiple_speakers():
    db = SpeakerDB()
    db.enroll_from_embedding("Alice", [1.0, 0.0])
    db.enroll_from_embedding("Bob", [0.0, 1.0])

    result = DiarizationResult(
        segments=[
            DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=5.0),
            DiarizationSegment(speaker="SPEAKER_01", start=5.0, end=10.0),
        ],
        num_speakers=2,
        embeddings={
            "SPEAKER_00": [0.9, 0.1],  # Alice
            "SPEAKER_01": [0.1, 0.9],  # Bob
        },
    )

    names = db.identify(result, threshold=0.5)
    assert names == {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}


def test_identify_no_embeddings():
    db = SpeakerDB()
    db.enroll_from_embedding("Alice", [1.0, 0.0])

    result = DiarizationResult(
        segments=[DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=5.0)],
        num_speakers=1,
    )

    names = db.identify(result)
    assert names == {}


def test_save_and_load(tmp_path: Path):
    db = SpeakerDB()
    db.enroll_from_embedding("Alice", [0.1, 0.2, 0.3])
    db.enroll_from_embedding("Bob", [0.4, 0.5, 0.6])

    path = tmp_path / "speakers.json"
    db.save(path)

    loaded = SpeakerDB.load(path)
    assert loaded.names == ["Alice", "Bob"]
    assert len(loaded) == 2
    assert loaded.profiles["Alice"].embedding == [0.1, 0.2, 0.3]


def test_repr():
    db = SpeakerDB()
    db.enroll_from_embedding("Alice", [1.0])
    assert "Alice" in repr(db)
