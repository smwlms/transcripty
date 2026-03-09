"""Benchmark transcripty diarization against Plaud speaker labels."""

import json
import logging
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

from transcripty import diarize, merge, transcribe  # noqa: E402

PLAUDE_ROOT = Path("/Users/samuelwillems/Documents/Projecten/Plaude")
DB_PATH = PLAUDE_ROOT / "plaude.db"
RECORDING_ID = 47


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


def main():
    ref = get_reference(RECORDING_ID)
    audio_path = PLAUDE_ROOT / ref["storage_path"]
    duration_s = ref["duration_ms"] / 1000
    model_size = "medium"

    print(f"{'=' * 70}")
    print(f"DIARIZATION BENCHMARK: {ref['filename']}")
    print(f"Audio: {audio_path.name} | Duration: {duration_s:.0f}s | Model: {model_size}")
    print(f"{'=' * 70}")

    # Step 1: Transcribe
    print("\n--- Step 1: Transcription ---")
    t0 = time.time()
    result = transcribe(
        audio_path=str(audio_path),
        model_size=model_size,
        language=None,
        word_timestamps=True,
        compute_type="int8",
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

    # Step 3: Merge
    print("\n--- Step 3: Merge ---")
    labeled = merge(result.segments, speakers.segments)

    # Output
    print(f"\n{'=' * 70}")
    print("TRANSCRIPTY OUTPUT (with speakers)")
    print(f"{'=' * 70}\n")
    for seg in labeled:
        print(f"[{fmt_ts(seg.start)}-{fmt_ts(seg.end)}] {seg.speaker}: {seg.text}")

    print(f"\n{'=' * 70}")
    print("PLAUD REFERENCE")
    print(f"{'=' * 70}\n")
    print(ref["transcription_text"][:3000])
    if len(ref["transcription_text"]) > 3000:
        print(f"\n... ({len(ref['transcription_text'])} chars total)")

    # Stats
    total_time = t_transcribe + t_diarize
    print(f"\n{'=' * 70}")
    print("STATS")
    print(f"{'=' * 70}")
    print(f"Transcription: {t_transcribe:.1f}s")
    print(f"Diarization:   {t_diarize:.1f}s")
    print(f"Total:         {total_time:.1f}s ({total_time / duration_s:.2f}x realtime)")
    print(f"Speakers:      {speakers.num_speakers}")
    print(f"Segments:      {len(labeled)}")

    # Save
    output = {
        "recording_id": RECORDING_ID,
        "filename": ref["filename"],
        "model": model_size,
        "transcribe_seconds": round(t_transcribe, 2),
        "diarize_seconds": round(t_diarize, 2),
        "total_seconds": round(total_time, 2),
        "realtime_factor": round(total_time / duration_s, 3),
        "num_speakers": speakers.num_speakers,
        "num_segments": len(labeled),
        "labeled_segments": [s.model_dump() for s in labeled],
        "diarization_segments": [s.model_dump() for s in speakers.segments],
        "reference_text": ref["transcription_text"],
    }
    out_path = Path(__file__).parent / "benchmark_diarize.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
