"""Full pipeline benchmark: vocabulary + transcription + diarization + speaker ID."""

import logging
import os
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

from transcripty import Vocabulary, diarize, merge, transcribe  # noqa: E402
from transcripty.speakers import SpeakerDB  # noqa: E402

PLAUDE_ROOT = Path(os.environ.get("PLAUDE_ROOT", Path.home() / "Documents/Projecten/Plaude"))
DB_PATH = PLAUDE_ROOT / "plaude.db"
RECORDING_ID = int(os.environ.get("RECORDING_ID", "47"))
SPEAKERS_FILE = Path(__file__).parent / "speakers.json"
VOCAB_FILE = Path(__file__).parent / "vocabulary.json"


def get_reference(recording_id: int) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT filename, duration_ms, storage_path, transcription_text "
        "FROM recordings WHERE id = ?",
        (recording_id,),
    ).fetchone()
    conn.close()
    return dict(row)


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def setup_vocabulary():
    """Create vocabulary with domain-specific terms."""
    vocab = Vocabulary(
        [
            "Claes & Willems",
            "inkopen",
            "inkoper",
            "Whise",
            "Colibry",
            "radiatoren",
            "radiator",
            "loodgieter",
            "offerte",
            "Andries",
            "zaakvoerder",
        ]
    )
    vocab.save(VOCAB_FILE)
    return vocab


def main():
    ref = get_reference(RECORDING_ID)
    audio_path = PLAUDE_ROOT / ref["storage_path"]
    duration_s = ref["duration_ms"] / 1000

    print(f"{'=' * 70}")
    print(f"FULL PIPELINE BENCHMARK: {ref['filename']}")
    print(f"Audio: {audio_path.name} | Duration: {duration_s:.0f}s")
    print(f"{'=' * 70}")

    # Setup vocabulary
    vocab = setup_vocabulary()
    print(f"\nVocabulary: {vocab.as_prompt()}")

    # Step 1: Transcribe with vocabulary prompt
    print("\n--- Step 1: Transcription (medium + vocabulary) ---")
    t0 = time.time()
    result = transcribe(
        audio_path=str(audio_path),
        model_size="medium",
        language=None,
        word_timestamps=True,
        compute_type="int8",
        prompt=vocab.as_prompt(),
    )
    t_transcribe = time.time() - t0
    print(f"Done in {t_transcribe:.1f}s — {len(result.segments)} segments")

    # Step 2: Diarize
    print("\n--- Step 2: Diarization ---")
    t0 = time.time()
    speakers = diarize(
        audio_path=str(audio_path),
        min_speakers=2,
        max_speakers=3,
    )
    t_diarize = time.time() - t0
    print(f"Done in {t_diarize:.1f}s — {speakers.num_speakers} speakers")
    print(f"Embeddings: {list(speakers.embeddings.keys())}")

    # Step 3: Try speaker identification (if profiles exist)
    speaker_names = {}
    if SPEAKERS_FILE.exists():
        db = SpeakerDB.load(SPEAKERS_FILE)
        speaker_names = db.identify(speakers)
        print(f"\nIdentified speakers: {speaker_names}")
    else:
        # Save embeddings for future enrollment
        db = SpeakerDB()
        for label, emb in speakers.embeddings.items():
            db.enroll_from_embedding(f"Unknown_{label}", emb)
        db.save(SPEAKERS_FILE)
        print(f"\nNo speaker profiles yet. Saved embeddings to {SPEAKERS_FILE}")
        print("Re-run after enrolling speakers to test identification.")

    # Step 4: Merge
    labeled = merge(result.segments, speakers.segments, speaker_names=speaker_names)

    # Output
    total_time = t_transcribe + t_diarize
    print(f"\n{'=' * 70}")
    print("TRANSCRIPTY OUTPUT (vocab + diarization)")
    print(f"{'=' * 70}\n")
    for seg in labeled:
        print(f"[{fmt_ts(seg.start)}-{fmt_ts(seg.end)}] {seg.speaker}: {seg.text}")

    # Compare key words that vocabulary should improve
    print(f"\n{'=' * 70}")
    print("VOCABULARY IMPACT CHECK")
    print(f"{'=' * 70}")
    full_text = " ".join(seg.text for seg in labeled)
    check_words = [
        ("Claes & Willems", "Klaas"),
        ("radiatoren", "radiateuren"),
        ("loodgieter", "luchtgieter"),
        ("offerte", "ofeste"),
        ("zaakvoerder", "zaak voelden"),
        ("Andries", None),
    ]
    for correct, wrong in check_words:
        found_correct = correct.lower() in full_text.lower()
        found_wrong = wrong and wrong.lower() in full_text.lower()
        if found_correct and not found_wrong:
            status = "OK"
        elif not found_correct:
            status = "MISS"
        else:
            status = "MIXED"
        found_str = "found" if found_correct else "NOT found"
        wrong_str = ""
        if wrong:
            present = "still present" if found_wrong else "gone"
            wrong_str = f" (wrong form '{wrong}' {present})"
        print(f"  {status:5} | '{correct}' {found_str}{wrong_str}")

    print(f"\n{'=' * 70}")
    print("STATS")
    print(f"{'=' * 70}")
    print(f"Transcription: {t_transcribe:.1f}s")
    print(f"Diarization:   {t_diarize:.1f}s")
    print(f"Total:         {total_time:.1f}s ({total_time / duration_s:.2f}x realtime)")


if __name__ == "__main__":
    main()
