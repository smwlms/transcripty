"""Benchmark transcripty against existing Plaud transcriptions."""

import json
import logging
import sqlite3
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

from transcripty import transcribe

# Paths
PLAUDE_ROOT = Path("/Users/samuelwillems/Documents/Projecten/Plaude")
DB_PATH = PLAUDE_ROOT / "plaude.db"

# Recording to test
RECORDING_ID = 47


def get_reference(recording_id: int) -> dict:
    """Get existing transcription from Plaude database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT filename, duration_ms, storage_path, transcription_text, "
        "transcription_language FROM recordings WHERE id = ?",
        (recording_id,),
    ).fetchone()
    conn.close()
    return dict(row)


def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main():
    # Load reference
    ref = get_reference(RECORDING_ID)
    audio_path = PLAUDE_ROOT / ref["storage_path"]

    print("=" * 70)
    print(f"BENCHMARK: {ref['filename']}")
    print(f"Audio: {audio_path.name} ({audio_path.stat().st_size / 1024:.0f} KB)")
    print(f"Duration: {ref['duration_ms'] / 1000:.0f}s")
    print("=" * 70)

    # Run transcription
    print("\n--- Running transcripty (model=small, compute=int8) ---\n")
    start = time.time()
    result = transcribe(
        audio_path=str(audio_path),
        model_size="small",
        language=None,  # auto-detect
        word_timestamps=True,
        compute_type="int8",
    )
    elapsed = time.time() - start

    # Results
    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"Time:     {elapsed:.1f}s (realtime factor: {elapsed / (ref['duration_ms'] / 1000):.2f}x)")
    print(f"Language: {result.language} ({result.language_probability:.0%})")
    print(f"Duration: {result.duration:.1f}s")
    print(f"Segments: {result.segments.__len__()}")
    print()

    # Transcripty output
    print("--- TRANSCRIPTY OUTPUT ---")
    print()
    full_text_lines = []
    for seg in result.segments:
        ts = f"[{format_timestamp(seg.start)}-{format_timestamp(seg.end)}]"
        line = f"{ts} {seg.text}"
        print(line)
        full_text_lines.append(seg.text)
    full_text = " ".join(full_text_lines)

    # Reference output
    print()
    print("--- PLAUD REFERENCE ---")
    print()
    print(ref["transcription_text"][:2000])
    if len(ref["transcription_text"]) > 2000:
        print(f"\n... ({len(ref['transcription_text'])} chars total)")

    # Save results for detailed comparison
    output = {
        "recording_id": RECORDING_ID,
        "filename": ref["filename"],
        "model": "small",
        "compute_type": "int8",
        "elapsed_seconds": round(elapsed, 2),
        "realtime_factor": round(elapsed / (ref["duration_ms"] / 1000), 3),
        "detected_language": result.language,
        "language_probability": round(result.language_probability, 3),
        "num_segments": len(result.segments),
        "transcripty_text": full_text,
        "reference_text": ref["transcription_text"],
        "segments": [s.model_dump() for s in result.segments],
    }

    out_path = Path(__file__).parent / "benchmark_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
