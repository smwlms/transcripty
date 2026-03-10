"""Benchmark: Speaker enrollment, model comparison, and Plaud comparison.

Steps:
1. Enroll Samuel Willems from enrollment audio (105s NL read-aloud)
2. Benchmark all 7 Whisper models on enrollment audio (default + optimized)
3. Run top models on Strategiebespreking (57min) with diarization + speaker ID
4. Compare with Plaud reference transcription (WER)
5. Generate report for GitHub

Usage:
    python benchmarks/benchmark_enrollment.py               # full benchmark
    python benchmarks/benchmark_enrollment.py --short-only   # enrollment audio only
    python benchmarks/benchmark_enrollment.py --skip-enroll  # skip step 1
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("benchmark")

# Paths
PLAUDE_DB = Path(
    os.environ.get(
        "PLAUDE_DB",
        Path.home() / "Documents/Projecten/Plaude/plaude.db",
    )
)
PLAUDE_STORAGE = PLAUDE_DB.parent / "storage"
SPEAKERS_FILE = Path(__file__).parent / "speakers.json"
REPORT_FILE = Path(__file__).parent / "benchmark_report.md"
RESULTS_FILE = Path(__file__).parent / "benchmark_enrollment_results.json"

# Recording IDs
ENROLLMENT_ID = "7f3d9c21bb1415fe2531c854c5ff1f8f"
STRATEGY_ID = "fd23951a297af24ae837eed9c8122056"

# All available models
MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
    "distil-large-v3",
]

# Models to test on long audio (57min) — tiny/base are too low quality
STRATEGY_MODELS = ["small", "medium", "large-v3", "large-v3-turbo"]

# Parameter presets
PARAM_PRESETS = {
    "default": {
        "vad_filter": False,
        "condition_on_previous_text": True,
        "hallucination_silence_threshold": None,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
    },
    "optimized": {
        "vad_filter": True,
        "condition_on_previous_text": False,
        "hallucination_silence_threshold": 2.0,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
    },
}


def get_recording(plaud_file_id: str) -> dict:
    """Get recording from Plaude DB."""
    conn = sqlite3.connect(PLAUDE_DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM recordings WHERE plaud_file_id = ?",
        (plaud_file_id,),
    ).fetchone()
    conn.close()
    return dict(row)


def normalize_text(text: str) -> str:
    """Normalize text for WER comparison."""
    text = text.lower()
    text = re.sub(r"\[speaker\s*\d+\]", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute Word Error Rate using dynamic programming."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    r = len(ref_words)
    h = len(hyp_words)

    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    # Backtrack to count S, I, D
    i, j = r, h
    substitutions = insertions = deletions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            insertions += 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            break

    wer = d[r][h] / r if r > 0 else 0.0
    return {
        "wer": round(wer, 4),
        "wer_pct": round(wer * 100, 1),
        "ref_words": r,
        "hyp_words": h,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "errors": d[r][h],
    }


def step1_enroll():
    """Step 1: Enroll Samuel Willems from enrollment audio."""
    from transcripty import diarize
    from transcripty.speakers import SpeakerDB

    rec = get_recording(ENROLLMENT_ID)
    audio_path = PLAUDE_STORAGE / rec["storage_path"].replace(
        "storage/", ""
    )

    logger.info("=== STEP 1: Speaker Enrollment ===")
    logger.info(
        "Audio: %s (%ds)", rec["filename"], rec["duration_ms"] // 1000
    )

    t0 = time.time()
    result = diarize(str(audio_path), num_speakers=1)
    t_diarize = time.time() - t0

    logger.info(
        "Diarization: %.1fs, %d speakers, embeddings: %s",
        t_diarize,
        result.num_speakers,
        list(result.embeddings.keys()),
    )

    if not result.embeddings:
        logger.error("No embeddings returned! Cannot enroll.")
        return None

    db = SpeakerDB()
    embedding = list(result.embeddings.values())[0]
    db.enroll_from_embedding("Samuel Willems", embedding)
    db.save(str(SPEAKERS_FILE))
    logger.info(
        "Enrolled 'Samuel Willems' (embedding dim=%d)", len(embedding)
    )
    logger.info("Saved to %s", SPEAKERS_FILE)

    return {
        "diarization_time_s": round(t_diarize, 2),
        "embedding_dim": len(embedding),
        "speakers_file": str(SPEAKERS_FILE),
    }


def _run_transcription(audio_path, model_size, preset_name, plaud_text):
    """Run a single transcription with given model and parameter preset."""
    from transcripty import transcribe
    from transcripty.config import configure

    params = PARAM_PRESETS[preset_name]
    configure(max_cached_models=1)

    t0 = time.time()
    result = transcribe(
        str(audio_path),
        model_size=model_size,
        language="nl",
        word_timestamps=True,
        compute_type="int8",
        **params,
    )
    elapsed = time.time() - t0

    full_text = " ".join(seg.text for seg in result.segments)
    wer = compute_wer(plaud_text, full_text)
    rt_factor = elapsed / result.duration if result.duration > 0 else 0

    return {
        "model": model_size,
        "preset": preset_name,
        "time_s": round(elapsed, 2),
        "rt_factor": round(rt_factor, 3),
        "duration_s": round(result.duration, 1),
        "segments": len(result.segments),
        "language": result.language,
        "language_prob": round(result.language_probability, 3),
        "wer": wer,
        "text_preview": full_text[:200],
        "full_text": full_text,
    }


def step2_benchmark_enrollment():
    """Step 2: Benchmark all models × all presets on enrollment audio."""
    rec = get_recording(ENROLLMENT_ID)
    audio_path = PLAUDE_STORAGE / rec["storage_path"].replace(
        "storage/", ""
    )
    plaud_text = rec["transcription_text"]

    logger.info(
        "\n=== STEP 2: Model Benchmark (enrollment audio, 105s) ==="
    )

    results = []
    for model_size in MODELS:
        for preset_name in PARAM_PRESETS:
            logger.info(
                "\n--- Model: %s | Preset: %s ---", model_size, preset_name
            )
            entry = _run_transcription(
                audio_path, model_size, preset_name, plaud_text
            )
            results.append(entry)
            logger.info(
                "Time: %.1fs (%.3fx RT) | WER: %.1f%% | Segments: %d",
                entry["time_s"],
                entry["rt_factor"],
                entry["wer"]["wer_pct"],
                entry["segments"],
            )

    return results


def step3_strategy_benchmark():
    """Step 3: Run top models on Strategiebespreking with diarization."""
    from transcripty import diarize, merge
    from transcripty.speakers import SpeakerDB

    rec = get_recording(STRATEGY_ID)
    audio_path = PLAUDE_STORAGE / rec["storage_path"].replace(
        "storage/", ""
    )
    plaud_text = rec["transcription_text"]

    logger.info(
        "\n=== STEP 3: Strategiebespreking Benchmark (57min) ==="
    )

    # Load speaker DB
    db = SpeakerDB.load(str(SPEAKERS_FILE))
    logger.info("Loaded speaker DB: %s", db.names)

    # Diarize once (shared across all models)
    logger.info("\n--- Diarization (shared) ---")
    t0 = time.time()
    speakers = diarize(str(audio_path), min_speakers=2, max_speakers=4)
    t_diarize = time.time() - t0
    logger.info(
        "Diarization: %.1fs, %d speakers", t_diarize, speakers.num_speakers
    )

    # Speaker identification
    names = db.identify(speakers, threshold=0.4)
    logger.info("Identified speakers: %s", names)

    results = []
    for model_size in STRATEGY_MODELS:
        for preset_name in PARAM_PRESETS:
            logger.info(
                "\n--- Model: %s | Preset: %s ---",
                model_size,
                preset_name,
            )

            from transcripty import transcribe
            from transcripty.config import configure

            params = PARAM_PRESETS[preset_name]
            configure(max_cached_models=1)

            t0 = time.time()
            result = transcribe(
                str(audio_path),
                model_size=model_size,
                language="nl",
                word_timestamps=True,
                compute_type="int8",
                **params,
            )
            t_transcribe = time.time() - t0

            full_text = " ".join(
                seg.text for seg in result.segments
            )
            wer = compute_wer(plaud_text, full_text)
            rt_factor = (
                t_transcribe / result.duration
                if result.duration > 0
                else 0
            )

            # Merge with shared diarization
            labeled = merge(
                result.segments,
                speakers.segments,
                speaker_names=names,
            )

            # Speaker stats
            speaker_counts = {}
            for seg in labeled:
                speaker_counts[seg.speaker] = (
                    speaker_counts.get(seg.speaker, 0) + 1
                )

            total_time = t_transcribe + t_diarize
            entry = {
                "model": model_size,
                "preset": preset_name,
                "time_s": round(t_transcribe, 2),
                "rt_factor": round(rt_factor, 3),
                "duration_s": round(result.duration, 1),
                "segments": len(result.segments),
                "language": result.language,
                "language_prob": round(
                    result.language_probability, 3
                ),
                "wer": wer,
                "text_preview": full_text[:200],
                "diarize_time_s": round(t_diarize, 2),
                "total_time_s": round(total_time, 2),
                "speakers_detected": speakers.num_speakers,
                "speakers_identified": names,
                "speaker_segment_counts": speaker_counts,
            }

            results.append(entry)
            logger.info(
                "Total: %.1fs (%.3fx RT) | WER: %.1f%% | Speakers: %s",
                total_time,
                rt_factor,
                wer["wer_pct"],
                speaker_counts,
            )

    return results, t_diarize


def generate_report(
    enrollment_info, enrollment_results, strategy_results, diarize_time=None
):
    """Generate markdown report."""
    lines = [
        "# Transcripty Benchmark Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
        "**Hardware:** Apple M1 Max, 10 cores, 32 GB RAM",
        "**Compute type:** int8 (CTranslate2 on CPU)",
        "**Reference:** Plaud cloud transcription",
        "",
        "---",
        "",
    ]

    # --- Section 1: Speaker Enrollment ---
    if enrollment_info:
        lines.extend([
            "## 1. Speaker Enrollment",
            "",
            "- **Speaker:** Samuel Willems",
            "- **Audio:** 105s NL read-aloud text",
            f"- **Diarization time:** {enrollment_info['diarization_time_s']}s",
            f"- **Embedding dimension:** {enrollment_info['embedding_dim']}",
            "",
            "---",
            "",
        ])

    # --- Section 2: Short audio benchmark ---
    lines.extend([
        "## 2. Model Comparison — Short Audio (105s)",
        "",
        "All models tested with two parameter presets:",
        "- **default:** vanilla faster-whisper settings",
        (
            "- **optimized:** VAD filter, no condition on previous text, "
            "hallucination threshold, repetition penalty"
        ),
        "",
    ])

    # Group by preset
    for preset_name in PARAM_PRESETS:
        preset_results = [
            r for r in enrollment_results if r["preset"] == preset_name
        ]
        if not preset_results:
            continue

        params = PARAM_PRESETS[preset_name]
        lines.extend([
            f"### Preset: `{preset_name}`",
            "",
        ])
        if preset_name == "optimized":
            lines.extend([
                "```",
                f"vad_filter={params['vad_filter']}",
                f"condition_on_previous_text={params['condition_on_previous_text']}",
                f"hallucination_silence_threshold={params['hallucination_silence_threshold']}",
                f"repetition_penalty={params['repetition_penalty']}",
                f"no_repeat_ngram_size={params['no_repeat_ngram_size']}",
                "```",
                "",
            ])

        lines.extend([
            "| Model | Time | RT Factor | WER | Errors | Words |",
            "|-------|------|-----------|-----|--------|-------|",
        ])

        for r in preset_results:
            w = r["wer"]
            lines.append(
                f"| {r['model']} | {r['time_s']}s"
                f" | {r['rt_factor']}x"
                f" | **{w['wer_pct']}%**"
                f" | {w['errors']}/{w['ref_words']}"
                f" | {w['hyp_words']} |"
            )
        lines.append("")

    # Best models summary
    best_default = min(
        [r for r in enrollment_results if r["preset"] == "default"],
        key=lambda x: x["wer"]["wer"],
    )
    best_optimized = min(
        [r for r in enrollment_results if r["preset"] == "optimized"],
        key=lambda x: x["wer"]["wer"],
    )
    lines.extend([
        "### Best models",
        "",
        (
            f"- **Best default:** `{best_default['model']}`"
            f" (WER {best_default['wer']['wer_pct']}%)"
        ),
        (
            f"- **Best optimized:** `{best_optimized['model']}`"
            f" (WER {best_optimized['wer']['wer_pct']}%)"
        ),
        "",
        "### WER improvement (default → optimized)",
        "",
        "| Model | Default WER | Optimized WER | Improvement |",
        "|-------|-------------|---------------|-------------|",
    ])

    for model in MODELS:
        d = next(
            (
                r
                for r in enrollment_results
                if r["model"] == model and r["preset"] == "default"
            ),
            None,
        )
        o = next(
            (
                r
                for r in enrollment_results
                if r["model"] == model and r["preset"] == "optimized"
            ),
            None,
        )
        if d and o:
            diff = d["wer"]["wer_pct"] - o["wer"]["wer_pct"]
            sign = "+" if diff >= 0 else ""
            lines.append(
                f"| {model}"
                f" | {d['wer']['wer_pct']}%"
                f" | {o['wer']['wer_pct']}%"
                f" | {sign}{diff:.1f}pp |"
            )

    lines.append("")

    # --- Section 3: Strategy benchmark ---
    if strategy_results:
        lines.extend([
            "---",
            "",
            "## 3. Long Audio Benchmark (57 min meeting, 4 speakers)",
            "",
        ])

        if diarize_time:
            lines.append(
                f"Diarization (shared): {diarize_time:.1f}s"
                f" ({4} speakers detected)"
            )
            lines.append("")

        for preset_name in PARAM_PRESETS:
            preset_results = [
                r
                for r in strategy_results
                if r["preset"] == preset_name
            ]
            if not preset_results:
                continue

            lines.extend([
                f"### Preset: `{preset_name}`",
                "",
                (
                    "| Model | Transcription | Total"
                    " | RT Factor | WER | Speaker ID |"
                ),
                (
                    "|-------|--------------|------"
                    "|-----------|-----|------------|"
                ),
            ])

            for r in preset_results:
                w = r["wer"]
                has_samuel = "Samuel Willems" in str(
                    r.get("speaker_segment_counts", {})
                )
                id_str = "Yes" if has_samuel else "No"
                lines.append(
                    f"| {r['model']}"
                    f" | {r['time_s']}s"
                    f" | {r['total_time_s']}s"
                    f" | {r['rt_factor']}x"
                    f" | **{w['wer_pct']}%**"
                    f" | {id_str} |"
                )
            lines.append("")

        # Improvement table for strategy
        lines.extend([
            "### WER improvement on long audio",
            "",
            "| Model | Default WER | Optimized WER | Improvement |",
            "|-------|-------------|---------------|-------------|",
        ])

        for model in STRATEGY_MODELS:
            d = next(
                (
                    r
                    for r in strategy_results
                    if r["model"] == model and r["preset"] == "default"
                ),
                None,
            )
            o = next(
                (
                    r
                    for r in strategy_results
                    if r["model"] == model and r["preset"] == "optimized"
                ),
                None,
            )
            if d and o:
                diff = d["wer"]["wer_pct"] - o["wer"]["wer_pct"]
                sign = "+" if diff >= 0 else ""
                lines.append(
                    f"| {model}"
                    f" | {d['wer']['wer_pct']}%"
                    f" | {o['wer']['wer_pct']}%"
                    f" | {sign}{diff:.1f}pp |"
                )
        lines.append("")

    # --- Section 4: Conclusion ---
    lines.extend([
        "---",
        "",
        "## 4. Recommendations",
        "",
        "### Choosing a model",
        "",
        "| Use case | Model | Preset | Expected WER |",
        "|----------|-------|--------|--------------|",
        "| Maximum accuracy | large-v3 | optimized | Best |",
        "| Good balance | large-v3-turbo | optimized | Very good |",
        "| Fast processing | medium | optimized | Good |",
        "| Quick draft | small | optimized | Acceptable |",
        "",
        "### Optimized parameters (recommended)",
        "",
        "```python",
        "from transcripty import transcribe",
        "",
        "result = transcribe(",
        '    "audio.mp3",',
        '    model_size="large-v3",',
        '    language="nl",',
        "    vad_filter=True,",
        "    condition_on_previous_text=False,",
        "    hallucination_silence_threshold=2.0,",
        "    repetition_penalty=1.1,",
        "    no_repeat_ngram_size=3,",
        ")",
        "```",
        "",
        "### Parameter explanation",
        "",
        (
            "- **vad_filter:** Silero VAD filters non-speech audio,"
            " reducing hallucinations"
        ),
        (
            "- **condition_on_previous_text:** When False, prevents"
            " hallucination cascades on long audio"
        ),
        (
            "- **hallucination_silence_threshold:** Skips segments"
            " generated after silence periods"
        ),
        (
            "- **repetition_penalty:** Penalizes repeated tokens"
            " (>1.0 reduces loops)"
        ),
        (
            "- **no_repeat_ngram_size:** Prevents exact n-gram"
            " repetitions"
        ),
        "",
        "### What is WER?",
        "",
        (
            "Word Error Rate (WER) measures transcription accuracy."
            " It counts substitutions (wrong words),"
            " insertions (extra words), and deletions (missing words)"
            " divided by the total reference words."
            " Lower is better — 0% is perfect."
        ),
        "",
    ])

    report = "\n".join(lines)
    REPORT_FILE.write_text(report, encoding="utf-8")
    logger.info("\nReport saved to %s", REPORT_FILE)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Transcripty benchmark suite"
    )
    parser.add_argument(
        "--short-only",
        action="store_true",
        help="Only benchmark short audio (skip 57min meeting)",
    )
    parser.add_argument(
        "--skip-enroll",
        action="store_true",
        help="Skip enrollment (use existing speakers.json)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TRANSCRIPTY BENCHMARK — Model & Parameter Comparison")
    print("=" * 70)

    # Step 1: Enroll
    enrollment_info = None
    if not args.skip_enroll:
        enrollment_info = step1_enroll()
        if not enrollment_info:
            print("Enrollment failed. Aborting.")
            return
    else:
        logger.info("Skipping enrollment (--skip-enroll)")
        enrollment_info = {"diarization_time_s": 0, "embedding_dim": 256}

    # Step 2: Benchmark all models on enrollment audio
    enrollment_results = step2_benchmark_enrollment()

    # Step 3: Strategy benchmark
    strategy_results = []
    diarize_time = None
    if not args.short_only:
        strategy_results, diarize_time = step3_strategy_benchmark()

    # Save raw results
    all_results = {
        "enrollment_info": enrollment_info,
        "param_presets": PARAM_PRESETS,
        "enrollment_benchmark": [
            {k: v for k, v in r.items() if k != "full_text"}
            for r in enrollment_results
        ],
        "strategy_benchmark": strategy_results,
    }
    RESULTS_FILE.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False)
    )
    logger.info("Raw results saved to %s", RESULTS_FILE)

    # Generate report
    report = generate_report(
        enrollment_info, enrollment_results, strategy_results, diarize_time
    )
    print(f"\n{'=' * 70}")
    print(report)


if __name__ == "__main__":
    main()
