"""Adaptive speaker enrollment — improves voiceprints with each new sample.

Instead of averaging (which smooths out distinctive features for twins),
this module selects the MOST distinctive sample for each speaker by
maximizing inter-speaker ECAPA distance.

Usage:
    from transcripty.adaptive import select_best_enrollment
    best = select_best_enrollment(candidates, other_speakers)
"""

from __future__ import annotations

import logging
from pathlib import Path

from transcripty.speakers import _cosine_similarity

logger = logging.getLogger(__name__)


def select_best_enrollment(
    candidate_embeddings: list[list[float]],
    other_speaker_embeddings: list[list[float]],
) -> int:
    """Select the enrollment sample that maximizes distance from other speakers.

    For twins or speakers with similar voices, this picks the sample
    that is MOST distinctive compared to the other speakers in the DB.

    Args:
        candidate_embeddings: List of ECAPA embeddings for the target speaker's
            enrollment samples (e.g. 4 different recordings).
        other_speaker_embeddings: List of ECAPA embeddings for all OTHER speakers
            in the database that might be confused with this one.

    Returns:
        Index of the best candidate embedding (0-based).
    """
    if not candidate_embeddings:
        raise ValueError("No candidate embeddings provided")

    if not other_speaker_embeddings:
        return 0  # No other speakers to compare against

    best_idx = 0
    best_avg_distance = 0.0

    for i, candidate in enumerate(candidate_embeddings):
        # Average distance to all other speakers
        distances = [
            1 - _cosine_similarity(candidate, other)
            for other in other_speaker_embeddings
        ]
        avg_distance = sum(distances) / len(distances)

        if avg_distance > best_avg_distance:
            best_avg_distance = avg_distance
            best_idx = i

        logger.debug(
            "Candidate %d: avg distance = %.4f", i, avg_distance,
        )

    logger.info(
        "Best enrollment: candidate %d (avg distance = %.4f)",
        best_idx, best_avg_distance,
    )
    return best_idx


def update_speaker_with_best_sample(
    speaker_name: str,
    new_audio_path: str | Path,
    speaker_db_path: str | Path,
    confusable_names: list[str] | None = None,
) -> dict:
    """Re-evaluate which enrollment sample is best after adding a new one.

    Extracts ECAPA embedding from the new audio, compares with existing
    samples and confusable speakers, and updates the DB if the new sample
    provides better separation.

    Args:
        speaker_name: Name of the speaker to update.
        new_audio_path: Path to new audio sample.
        speaker_db_path: Path to the enhanced speaker DB JSON.
        confusable_names: Names of speakers that might be confused with this one.
            If None, compares against ALL other speakers.

    Returns:
        Dict with "updated" (bool), "old_distance", "new_distance".
    """
    from transcripty.ecapa import extract_ecapa_embedding
    from transcripty.speakers import SpeakerDB

    db = SpeakerDB.load(str(speaker_db_path))

    if speaker_name not in db.profiles:
        return {"error": f"Speaker '{speaker_name}' not in DB"}

    profile = db.profiles[speaker_name]
    if not profile.ecapa_embedding:
        return {"error": f"No ECAPA embedding for '{speaker_name}'"}

    # Extract ECAPA from new sample
    new_ecapa = extract_ecapa_embedding(str(new_audio_path))

    # Determine confusable speakers
    if confusable_names:
        others = [
            db.profiles[n].ecapa_embedding
            for n in confusable_names
            if n in db.profiles and db.profiles[n].ecapa_embedding
        ]
    else:
        others = [
            p.ecapa_embedding
            for name, p in db.profiles.items()
            if name != speaker_name and p.ecapa_embedding
        ]

    if not others:
        return {"updated": False, "reason": "no confusable speakers"}

    # Compare: current vs new
    current_avg_distance = sum(
        1 - _cosine_similarity(profile.ecapa_embedding, o) for o in others
    ) / len(others)

    new_avg_distance = sum(
        1 - _cosine_similarity(new_ecapa, o) for o in others
    ) / len(others)

    if new_avg_distance > current_avg_distance:
        profile.ecapa_embedding = new_ecapa
        db.save(str(speaker_db_path))
        logger.info(
            "Updated '%s' ECAPA: distance %.4f → %.4f",
            speaker_name, current_avg_distance, new_avg_distance,
        )
        return {
            "updated": True,
            "old_distance": round(current_avg_distance, 4),
            "new_distance": round(new_avg_distance, 4),
        }

    return {
        "updated": False,
        "old_distance": round(current_avg_distance, 4),
        "new_distance": round(new_avg_distance, 4),
        "reason": "current sample is more distinctive",
    }
