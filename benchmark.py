"""Benchmark transcripty against existing Plaud transcriptions."""

import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

from transcripty import transcribe  # noqa: E402

# Paths
PLAUDE_ROOT = Path("/Users/samuelwillems/Documents/Projecten/Plaude")
DB_PATH = PLAUDE_ROOT / "plaude.db"
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


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def run_benchmark(model_size: str) -> dict:
    ref = get_reference(RECORDING_ID)
    audio_path = PLAUDE_ROOT / ref["storage_path"]
    duration_s = ref["duration_ms"] / 1000

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK: {ref['filename']} — model={model_size}")
    print(f"Audio: {audio_path.name} ({audio_path.stat().st_size / 1024:.0f} KB)")
    print(f"Duration: {duration_s:.0f}s")
    print(f"{'=' * 70}\n")

    start = time.time()
    result = transcribe(
        audio_path=str(audio_path),
        model_size=model_size,
        language=None,
        word_timestamps=True,
        compute_type="int8",
    )
    elapsed = time.time() - start
    rt_factor = elapsed / duration_s

    full_text_lines = []
    for seg in result.segments:
        line = f"[{fmt_ts(seg.start)}-{fmt_ts(seg.end)}] {seg.text}"
        print(line)
        full_text_lines.append(seg.text)
    full_text = " ".join(full_text_lines)

    print(f"\n--- Stats ---")
    print(f"Time:     {elapsed:.1f}s ({rt_factor:.2f}x realtime)")
    print(f"Language: {result.language} ({result.language_probability:.0%})")
    print(f"Segments: {len(result.segments)}")

    output = {
        "recording_id": RECORDING_ID,
        "filename": ref["filename"],
        "model": model_size,
        "compute_type": "int8",
        "elapsed_seconds": round(elapsed, 2),
        "realtime_factor": round(rt_factor, 3),
        "detected_language": result.language,
        "language_probability": round(result.language_probability, 3),
        "num_segments": len(result.segments),
        "transcripty_text": full_text,
        "reference_text": ref["transcription_text"],
        "segments": [s.model_dump() for s in result.segments],
    }

    out_path = Path(__file__).parent / f"benchmark_{model_size}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}\n")

    return output


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["small"]

    results = {}
    for model in models:
        results[model] = run_benchmark(model)

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print("COMPARISON")
        print(f"{'=' * 70}")
        print(f"{'Model':<12} {'Time':>8} {'RT Factor':>12} {'Segments':>10} {'Lang':>6}")
        print("-" * 50)
        for m, r in results.items():
            print(
                f"{m:<12} {r['elapsed_seconds']:>7.1f}s "
                f"{r['realtime_factor']:>11.3f}x "
                f"{r['num_segments']:>9} "
                f"{r['detected_language']:>5}"
            )


if __name__ == "__main__":
    main()
