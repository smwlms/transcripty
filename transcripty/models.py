"""Pydantic models for transcription, diarization, and merge results."""

from __future__ import annotations

from pydantic import BaseModel


class Word(BaseModel):
    """A single word with timing information."""

    text: str
    start: float
    end: float
    probability: float = 0.0


class Segment(BaseModel):
    """A transcription segment (phrase/sentence)."""

    text: str
    start: float
    end: float
    words: list[Word] = []


class TranscriptionResult(BaseModel):
    """Result of a transcription run."""

    segments: list[Segment]
    language: str
    language_probability: float
    duration: float


class DiarizationSegment(BaseModel):
    """A speaker segment from diarization."""

    speaker: str
    start: float
    end: float


class DiarizationResult(BaseModel):
    """Result of a diarization run."""

    segments: list[DiarizationSegment]
    num_speakers: int
    embeddings: dict[str, list[float]] = {}  # speaker_label -> embedding vector


class LabeledSegment(BaseModel):
    """A transcription segment with an assigned speaker label."""

    text: str
    start: float
    end: float
    speaker: str
    words: list[Word] = []
