"""WER Benchmark — Word Error Rate measurement against ground truth.

Reads ground_truth.json, transcribes each sample with given config,
computes WER, and reports per-sample and aggregate results.

Usage:
    PYTHONPATH=. .venv/bin/python benchmarks/wer_benchmark.py
    PYTHONPATH=. .venv/bin/python benchmarks/wer_benchmark.py --config speed
    PYTHONPATH=. .venv/bin/python benchmarks/wer_benchmark.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ECHO_ROOT = Path.home() / "Documents/Projecten/Plaude"
GT_PATH = Path(__file__).parent / "ground_truth.json"
RESULTS_PATH = Path(__file__).parent / "wer_results.json"


def levenshtein_words(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """Word-level Levenshtein distance with S/I/D counts."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = (distance, substitutions, insertions, deletions)
    d = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = (i, 0, 0, i)
    for j in range(1, m + 1):
        d[0][j] = (j, 0, j, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                sub = d[i - 1][j - 1]
                ins = d[i][j - 1]
                dele = d[i - 1][j]
                candidates = [
                    (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),  # substitution
                    (ins[0] + 1, ins[1], ins[2] + 1, ins[3]),  # insertion
                    (dele[0] + 1, dele[1], dele[2], dele[3] + 1),  # deletion
                ]
                d[i][j] = min(candidates, key=lambda x: x[0])

    return d[n][m]


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute WER with detailed error breakdown."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return {
            "wer": 0.0 if not hyp_words else 1.0,
            "wer_pct": 0.0 if not hyp_words else 100.0,
            "ref_words": 0,
            "hyp_words": len(hyp_words),
            "errors": len(hyp_words),
            "substitutions": 0,
            "insertions": len(hyp_words),
            "deletions": 0,
        }

    dist, subs, ins, dels = levenshtein_words(ref_words, hyp_words)
    wer = dist / len(ref_words)

    return {
        "wer": round(wer, 4),
        "wer_pct": round(wer * 100, 1),
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "errors": dist,
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
    }


def find_word_diffs(reference: str, hypothesis: str, max_diffs: int = 20) -> list[dict]:
    """Find specific word-level differences."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    diffs = []

    # Simple aligned comparison (not perfect for insertions/deletions)
    for i, (rw, hw) in enumerate(zip(ref_words, hyp_words)):
        if rw != hw:
            diffs.append({"pos": i, "ref": rw, "hyp": hw})
            if len(diffs) >= max_diffs:
                break

    if len(ref_words) != len(hyp_words):
        diffs.append({
            "pos": -1,
            "ref": f"({len(ref_words)} words)",
            "hyp": f"({len(hyp_words)} words)",
        })

    return diffs


# Preset configs
CONFIGS = {
    "p2_quality": dict(
        model_size="large-v3-turbo",
        compute_type="int8",
        beam_size=3,
        word_timestamps=True,
        temperature=0.0,
        vad_filter=True,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
        no_repeat_ngram_size=3,
        language=None,
        multilingual=True,
    ),
    "p3_balanced": dict(
        model_size="large-v3-turbo",
        compute_type="int8",
        beam_size=1,
        word_timestamps=True,
        temperature=(0.0, 0.2, 0.4),
        vad_filter=True,
        condition_on_previous_text=False,
        no_repeat_ngram_size=3,
        language="nl",
        multilingual=False,
    ),
    "p2_no_vad": dict(
        model_size="large-v3-turbo",
        compute_type="int8",
        beam_size=3,
        word_timestamps=True,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
        no_repeat_ngram_size=3,
        language=None,
        multilingual=True,
    ),
}


def run_benchmark(config_name: str = "p2_quality", output_json: bool = False):
    """Run WER benchmark on all ground truth samples."""
    from transcripty import transcribe

    if not GT_PATH.exists():
        print(f"Ground truth niet gevonden: {GT_PATH}")
        return

    with open(GT_PATH) as f:
        gt_data = json.load(f)

    config = CONFIGS.get(config_name, CONFIGS["p2_quality"])
    samples = [s for s in gt_data["samples"] if s.get("ground_truth")]

    if not samples:
        print("Geen samples met ground truth gevonden.")
        return

    print(f"\n{'═' * 60}")
    print(f"WER Benchmark — config: {config_name}")
    print(f"Samples: {len(samples)} met ground truth")
    print(f"{'═' * 60}")

    results = []
    total_ref_words = 0
    total_errors = 0

    for sample in samples:
        audio_path = ECHO_ROOT / sample["path"]
        if not audio_path.exists():
            print(f"\n  SKIP {sample['id']} — niet gevonden")
            continue

        print(f"\n  {sample['id']} ({sample['duration_s']}s, {sample['emotion']})...")
        t0 = time.time()
        r = transcribe(str(audio_path), **config)
        dt = time.time() - t0

        hyp_text = " ".join(seg.text for seg in r.segments)
        wer = compute_wer(sample["ground_truth"], hyp_text)
        diffs = find_word_diffs(sample["ground_truth"], hyp_text)

        rtf = round(dt / sample["duration_s"], 3)
        total_ref_words += wer["ref_words"]
        total_errors += wer["errors"]

        result = {
            "id": sample["id"],
            "config": config_name,
            "wer": wer,
            "rtf": rtf,
            "hypothesis": hyp_text,
            "diffs": diffs,
        }
        results.append(result)

        print(f"    WER={wer['wer_pct']}% ({wer['errors']}/{wer['ref_words']})")
        print(f"    S={wer['substitutions']} I={wer['insertions']} D={wer['deletions']}")
        print(f"    RTF={rtf}x")
        if diffs:
            for d in diffs[:5]:
                if d["pos"] >= 0:
                    print(f"    [{d['pos']}] {d['ref']} → {d['hyp']}")

    # Aggregate
    avg_wer = round(total_errors / total_ref_words * 100, 1) if total_ref_words > 0 else 0

    print(f"\n{'─' * 60}")
    print(f"GEMIDDELDE WER: {avg_wer}% ({total_errors}/{total_ref_words} errors)")
    print(f"Config: {config_name}")
    print(f"{'─' * 60}")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config_name,
        "config_params": config,
        "avg_wer": avg_wer,
        "total_ref_words": total_ref_words,
        "total_errors": total_errors,
        "samples": results,
    }

    if output_json:
        print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    else:
        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResultaten: {RESULTS_PATH}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="p2_quality", choices=list(CONFIGS.keys()))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    run_benchmark(args.config, args.json)
