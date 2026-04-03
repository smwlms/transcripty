"""Autoresearch measurement runner for Transcripty.

Measures RTF and hallucination_rate on a set of test audio files.
Used by the autoresearch loop to evaluate experiments.

Usage:
    python benchmarks/autoresearch_runner.py
    python benchmarks/autoresearch_runner.py --temperature 0.2 --vad_filter
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

ECHO_ROOT = Path.home() / "Documents/Projecten/Plaude"

TEST_FILES = [
    # (recording_id, path, duration_s, label)
    (53, ECHO_ROOT / "storage/2/e20d4d395b1d784ce6b6e4994c23887b.mp3", 191, "Roos gesprek NL"),
    (47, ECHO_ROOT / "storage/2/3e156ded9f471f1297962d37dce3c845.mp3", 313, "audio NL"),
    (22, ECHO_ROOT / "storage/2/0d1e65716cc93a4721e37e76683dacbb.mp3", 585, "audio4 NL"),
]

# Known Whisper hallucination patterns
HALLUCINATION_PATTERNS = [
    r"^\s*(Bedankt voor het kijken|Thanks for watching|Dank u voor het kijken)\s*\.?\s*$",
    r"^\s*(Ondertiteld door|Subtitles by|Ondertiteling)\b",
    r"^\s*\[Muziek\]\s*$",
    r"^\s*\[Music\]\s*$",
    r"^\s*\[Applaus\]\s*$",
    r"^\s*\[Applause\]\s*$",
    r"^\s*(\.{3,}|…)\s*$",
]
_halluc_re = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS]


def is_hallucination(text: str) -> bool:
    """Detect known hallucination patterns."""
    for pattern in _halluc_re:
        if pattern.match(text.strip()):
            return True
    return False


def count_repetitions(segments) -> int:
    """Count consecutive duplicate or near-duplicate segments."""
    count = 0
    for i in range(1, len(segments)):
        prev = segments[i - 1].text.strip().lower()
        curr = segments[i].text.strip().lower()
        if prev == curr and len(prev) > 5:
            count += 1
        # Near duplicate: one contains the other
        elif len(prev) > 10 and len(curr) > 10:
            if prev in curr or curr in prev:
                count += 1
    return count


def measure(audio_path: Path, duration_s: float, **kwargs) -> dict:
    """Run transcription and measure RTF + hallucination metrics."""
    from transcripty import transcribe

    start = time.time()
    result = transcribe(str(audio_path), **kwargs)
    elapsed = time.time() - start

    rtf = elapsed / duration_s

    segs = result.segments
    n_segs = len(segs)

    n_halluc = sum(1 for s in segs if is_hallucination(s.text))
    n_reps = count_repetitions(segs)
    n_bad = n_halluc + n_reps

    halluc_rate = n_bad / n_segs if n_segs > 0 else 0.0

    return {
        "rtf": round(rtf, 4),
        "segments": n_segs,
        "hallucinations": n_halluc,
        "repetitions": n_reps,
        "halluc_rate": round(halluc_rate, 4),
        "elapsed_s": round(elapsed, 2),
        "duration_s": duration_s,
    }


def run_suite(label: str, **kwargs) -> dict:
    """Run all test files and return aggregated results."""
    print(f"\n{'=' * 60}")
    print(f"Config: {label}")
    print(f"Params: {kwargs}")
    print(f"{'=' * 60}")

    results = []
    for rec_id, path, duration_s, name in TEST_FILES:
        if not path.exists():
            print(f"  SKIP {name} — bestand niet gevonden: {path}")
            continue
        print(f"\n  [{name}] {path.name} ({duration_s}s)")
        r = measure(path, duration_s, **kwargs)
        results.append(r)
        print(f"    RTF={r['rtf']:.3f}x  halluc={r['halluc_rate']:.1%}  segs={r['segments']}  "
              f"(halluc={r['hallucinations']} reps={r['repetitions']})")

    if not results:
        return {"error": "geen bestanden gevonden"}

    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    avg_halluc = sum(r["halluc_rate"] for r in results) / len(results)

    # combined_score = (1/rtf) * (1 - halluc_rate) — hoger = beter
    speed_score = 1 / avg_rtf if avg_rtf > 0 else 0
    quality_score = 1 - avg_halluc
    combined = round(speed_score * quality_score, 4)

    print(f"\n  GEMIDDELD: RTF={avg_rtf:.3f}x  halluc={avg_halluc:.1%}  combined={combined:.4f}")

    return {
        "label": label,
        "params": kwargs,
        "avg_rtf": round(avg_rtf, 4),
        "avg_halluc_rate": round(avg_halluc, 4),
        "combined_score": combined,
        "n_files": len(results),
        "details": results,
    }


# Baseline config (known optimal from Echo autoresearch)
BASELINE = dict(
    model_size="large-v3-turbo",
    compute_type="int8",
    beam_size=1,
    word_timestamps=True,
    temperature=0.0,
    vad_filter=False,
    condition_on_previous_text=True,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="custom")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--vad_filter", action="store_true", default=None)
    parser.add_argument("--no_vad_filter", action="store_true")
    parser.add_argument("--condition_on_previous_text", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--hallucination_silence_threshold", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    args = parser.parse_args()

    config = dict(BASELINE)
    if args.temperature is not None:
        config["temperature"] = args.temperature
    if args.vad_filter:
        config["vad_filter"] = True
    if args.no_vad_filter:
        config["vad_filter"] = False
    if args.condition_on_previous_text is not None:
        config["condition_on_previous_text"] = args.condition_on_previous_text
    if args.hallucination_silence_threshold is not None:
        config["hallucination_silence_threshold"] = args.hallucination_silence_threshold
    if args.repetition_penalty is not None:
        config["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size is not None:
        config["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    if args.beam_size is not None:
        config["beam_size"] = args.beam_size

    result = run_suite(args.label, **config)
    import json
    print(f"\nJSON: {json.dumps(result, indent=2)}")
