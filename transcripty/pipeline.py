"""High-level convenience pipeline combining transcription + diarization."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from transcripty.merge import merge
from transcripty.models import LabeledSegment
from transcripty.transcribe import transcribe

if TYPE_CHECKING:
    from transcripty.speakers import SpeakerDB

logger = logging.getLogger(__name__)


def _make_stage_callback(
    on_progress: Callable[[float, str], None],
    start: float,
    end: float,
) -> Callable[[float, str], None]:
    """Create a callback that maps 0.0-1.0 progress to a sub-range."""
    span = end - start

    def callback(progress: float, message: str) -> None:
        on_progress(start + progress * span, message)

    return callback


def transcribe_with_speakers(
    audio_path: str | Path,
    hf_token: str | None = None,
    num_speakers: int | None = None,
    speaker_db: SpeakerDB | None = None,
    speaker_threshold: float = 0.5,
    word_language_detection: bool = False,
    on_progress: Callable[[float, str], None] | None = None,
    **transcribe_kwargs,
) -> list[LabeledSegment]:
    """Transcribe audio and assign speaker labels in one call.

    Runs transcription, diarization, optional speaker identification,
    and merges the results.

    Args:
        audio_path: Path to the audio file.
        hf_token: HuggingFace token for diarization.
        num_speakers: Expected number of speakers (None for auto).
        speaker_db: Optional SpeakerDB instance for speaker identification.
        speaker_threshold: Minimum cosine similarity for speaker identification (0-1).
            Passed to SpeakerDB.identify(). Default 0.5.
        word_language_detection: When True, detect language per segment and
            retranscribe French/English segments in their correct language.
            Each Word gets a `language` field. Requires lingua-language-detector.
            Slower: adds one Whisper pass per non-Dutch segment.
        on_progress: Optional callback ``(progress: float, message: str) -> None``.
            Progress is 0.0–1.0, weighted across stages:
            0.0–0.6 transcription, 0.6–0.8 diarization, 0.8–0.9 merge/identify,
            0.9–1.0 word language detection (only if word_language_detection=True).
        **transcribe_kwargs: Additional arguments passed to transcribe().

    Returns:
        List of LabeledSegment with text, timestamps, and speaker labels.
        If word_language_detection=True, every Word also has a `language` field.
    """
    from transcripty.diarize import diarize

    # Build stage callbacks that map to weighted progress ranges
    transcribe_cb = None
    diarize_cb = None
    if on_progress:
        transcribe_end = 0.6 if word_language_detection else 0.7
        transcribe_cb = _make_stage_callback(on_progress, 0.0, transcribe_end)
        diarize_cb = _make_stage_callback(on_progress, transcribe_end, transcribe_end + 0.2)

    result = transcribe(audio_path, on_progress=transcribe_cb, **transcribe_kwargs)

    speakers = diarize(
        audio_path,
        hf_token=hf_token,
        num_speakers=num_speakers,
        on_progress=diarize_cb,
    )

    if on_progress:
        on_progress(0.85, "Merging segments...")

    names = None
    if speaker_db is not None:
        # Check if any profile has ECAPA embeddings for enhanced identification
        has_ecapa = any(p.ecapa_embedding for p in speaker_db.profiles.values())

        ecapa_embs = None
        prosodic_vecs = None

        if has_ecapa:
            try:
                from transcripty.ecapa import extract_ecapa_embeddings_for_segments
                from transcripty.prosody import extract_prosodic_for_segments

                # Build speaker_segments from diarization
                speaker_segs: dict[str, list[tuple[float, float]]] = {}
                for seg in speakers.segments:
                    speaker_segs.setdefault(seg.speaker, []).append((seg.start, seg.end))

                if on_progress:
                    on_progress(0.87, "Extracting ECAPA embeddings...")
                ecapa_embs = extract_ecapa_embeddings_for_segments(
                    audio_path, speaker_segs,
                )

                if on_progress:
                    on_progress(0.89, "Extracting prosodic features...")
                prosodic_feats = extract_prosodic_for_segments(audio_path, speaker_segs)
                prosodic_vecs = {
                    spk: f.as_vector() for spk, f in prosodic_feats.items()
                }

                logger.info(
                    "Enhanced speaker ID: ECAPA for %d speakers, prosody for %d",
                    len(ecapa_embs), len(prosodic_vecs),
                )
            except Exception as e:
                logger.warning("Enhanced speaker ID failed, falling back to pyannote-only: %s", e)

        names = speaker_db.identify(
            speakers,
            threshold=speaker_threshold,
            ecapa_embeddings=ecapa_embs,
            prosodic_features=prosodic_vecs,
        )

    labeled = merge(result.segments, speakers.segments, speaker_names=names)

    if word_language_detection and labeled:
        if on_progress:
            on_progress(0.9, "Detecting word languages (lingua fallback)...")
        try:
            from transcripty.language_detect import detect_segment_languages

            detect_segment_languages(labeled)
        except Exception as e:
            logger.warning("Lingua segment language detection failed: %s", e)
    elif transcribe_kwargs.get("multilingual") and labeled:
        # Fallback: segment-level lingua detection without retranscription
        try:
            from transcripty.language_detect import detect_segment_languages

            detect_segment_languages(labeled)
        except Exception as e:
            logger.warning("Per-segment language detection failed in pipeline: %s", e)

    if on_progress:
        on_progress(1.0, "Complete")

    return labeled
