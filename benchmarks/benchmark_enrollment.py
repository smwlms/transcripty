"""Benchmark: Speaker enrollment, model comparison, and Plaud comparison.

Steps:
1. Enroll Samuel Willems from enrollment audio (105s NL read-aloud)
2. Benchmark all 6 Whisper models on enrollment audio
3. Run medium + large-v3 on Strategiebespreking (57min) with diarization + speaker ID
4. Compare with Plaud reference transcription (WER)
5. Generate report
"""

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
PLAUDE_DB = Path(os.environ.get(
    "PLAUDE_DB",
    Path.home() / "Documents/Projecten/Plaude/plaude.db",
))
PLAUDE_STORAGE = PLAUDE_DB.parent / "storage"
SPEAKERS_FILE = Path(__file__).parent / "speakers.json"
REPORT_FILE = Path(__file__).parent / "benchmark_report.md"
RESULTS_FILE = Path(__file__).parent / "benchmark_enrollment_results.json"

# Recording IDs
ENROLLMENT_ID = "7f3d9c21bb1415fe2531c854c5ff1f8f"
STRATEGY_ID = "fd23951a297af24ae837eed9c8122056"

MODELS = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]
STRATEGY_MODELS = ["medium", "large-v3"]


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
    # Remove speaker labels like [Speaker 1]
    text = re.sub(r"\[speaker\s*\d+\]", "", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute Word Error Rate using dynamic programming."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    r = len(ref_words)
    h = len(hyp_words)

    # DP matrix
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
    audio_path = PLAUDE_STORAGE / rec["storage_path"].replace("storage/", "")

    logger.info("=== STEP 1: Speaker Enrollment ===")
    logger.info("Audio: %s (%ds)", rec["filename"], rec["duration_ms"] // 1000)

    t0 = time.time()
    result = diarize(str(audio_path), num_speakers=1)
    t_diarize = time.time() - t0

    logger.info("Diarization: %.1fs, %d speakers, embeddings: %s",
                t_diarize, result.num_speakers, list(result.embeddings.keys()))

    if not result.embeddings:
        logger.error("No embeddings returned! Cannot enroll.")
        return None

    db = SpeakerDB()
    embedding = list(result.embeddings.values())[0]
    db.enroll_from_embedding("Samuel Willems", embedding)
    db.save(str(SPEAKERS_FILE))
    logger.info("Enrolled 'Samuel Willems' (embedding dim=%d)", len(embedding))
    logger.info("Saved to %s", SPEAKERS_FILE)

    return {
        "diarization_time_s": round(t_diarize, 2),
        "embedding_dim": len(embedding),
        "speakers_file": str(SPEAKERS_FILE),
    }


def step2_benchmark_enrollment():
    """Step 2: Benchmark all models on enrollment audio."""
    from transcripty import transcribe
    from transcripty.config import configure

    rec = get_recording(ENROLLMENT_ID)
    audio_path = PLAUDE_STORAGE / rec["storage_path"].replace("storage/", "")
    plaud_text = rec["transcription_text"]

    logger.info("\n=== STEP 2: Model Benchmark (enrollment audio, 105s) ===")

    results = []
    for model_size in MODELS:
        logger.info("\n--- Model: %s ---", model_size)
        configure(max_cached_models=1)

        t0 = time.time()
        result = transcribe(
            str(audio_path),
            model_size=model_size,
            language="nl",
            word_timestamps=True,
            compute_type="int8",
        )
        elapsed = time.time() - t0

        full_text = " ".join(seg.text for seg in result.segments)
        wer = compute_wer(plaud_text, full_text)
        rt_factor = elapsed / result.duration if result.duration > 0 else 0

        entry = {
            "model": model_size,
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
        results.append(entry)

        logger.info("Time: %.1fs (%.3fx RT) | WER: %.1f%% | Segments: %d",
                     elapsed, rt_factor, wer["wer_pct"], len(result.segments))

    return results


def step3_strategy_benchmark():
    """Step 3: Run medium + large-v3 on Strategiebespreking with diarization."""
    from transcripty import diarize, merge, transcribe
    from transcripty.config import configure
    from transcripty.speakers import SpeakerDB

    rec = get_recording(STRATEGY_ID)
    audio_path = PLAUDE_STORAGE / rec["storage_path"].replace("storage/", "")
    plaud_text = rec["transcription_text"]

    logger.info("\n=== STEP 3: Strategiebespreking Benchmark (57min) ===")

    # Load speaker DB
    db = SpeakerDB.load(str(SPEAKERS_FILE))
    logger.info("Loaded speaker DB: %s", db.names)

    results = []
    for model_size in STRATEGY_MODELS:
        logger.info("\n--- Model: %s ---", model_size)
        configure(max_cached_models=1)

        # Transcribe
        t0 = time.time()
        result = transcribe(
            str(audio_path),
            model_size=model_size,
            language="nl",
            word_timestamps=True,
            compute_type="int8",
        )
        t_transcribe = time.time() - t0
        logger.info("Transcription: %.1fs", t_transcribe)

        # Diarize
        t0 = time.time()
        speakers = diarize(str(audio_path), min_speakers=2, max_speakers=4)
        t_diarize = time.time() - t0
        logger.info("Diarization: %.1fs, %d speakers", t_diarize, speakers.num_speakers)

        # Speaker identification
        names = db.identify(speakers, threshold=0.4)
        logger.info("Identified speakers: %s", names)

        # Merge
        labeled = merge(result.segments, speakers.segments, speaker_names=names)

        full_text = " ".join(seg.text for seg in result.segments)
        wer = compute_wer(plaud_text, full_text) if plaud_text else None

        # Speaker stats
        speaker_counts = {}
        for seg in labeled:
            speaker_counts[seg.speaker] = speaker_counts.get(seg.speaker, 0) + 1

        total_time = t_transcribe + t_diarize
        rt_factor = total_time / result.duration if result.duration > 0 else 0

        entry = {
            "model": model_size,
            "transcribe_time_s": round(t_transcribe, 2),
            "diarize_time_s": round(t_diarize, 2),
            "total_time_s": round(total_time, 2),
            "rt_factor": round(rt_factor, 3),
            "duration_s": round(result.duration, 1),
            "segments": len(result.segments),
            "speakers_detected": speakers.num_speakers,
            "speakers_identified": names,
            "speaker_segment_counts": speaker_counts,
            "wer": wer,
            "text_preview": full_text[:300],
        }
        results.append(entry)

        logger.info("Total: %.1fs (%.3fx RT) | WER: %.1f%% | Speakers: %s",
                     total_time, rt_factor,
                     wer["wer_pct"] if wer else -1,
                     speaker_counts)

    return results


def generate_report(enrollment_info, enrollment_results, strategy_results):
    """Generate markdown report."""
    lines = [
        "# Transcripty Benchmark Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Reference:** Plaud transcription service (~1 min processing)",
        "",
        "---",
        "",
        "## 1. Speaker Enrollment",
        "",
        f"- **Speaker:** Samuel Willems",
        f"- **Audio:** 2026-03-10 07:49:20 (105s, NL voorleestekst)",
        f"- **Diarization time:** {enrollment_info['diarization_time_s']}s",
        f"- **Embedding dimension:** {enrollment_info['embedding_dim']}",
        f"- **Saved to:** `speakers.json`",
        "",
        "---",
        "",
        "## 2. Model Comparison — Enrollment Audio (105s)",
        "",
        "Vergelijking van alle 6 Whisper modellen op de enrollment audio, met Plaud als referentie.",
        "",
        "| Model | Tijd | RT Factor | WER | Fouten | Woorden |",
        "|-------|------|-----------|-----|--------|---------|",
    ]

    for r in enrollment_results:
        w = r["wer"]
        lines.append(
            f"| {r['model']} | {r['time_s']}s | {r['rt_factor']}x | "
            f"**{w['wer_pct']}%** | {w['errors']}/{w['ref_words']} | "
            f"{w['hyp_words']} |"
        )

    # Best model
    best = min(enrollment_results, key=lambda x: x["wer"]["wer"])
    fastest = min(enrollment_results, key=lambda x: x["time_s"])

    lines.extend([
        "",
        f"**Beste kwaliteit:** `{best['model']}` (WER {best['wer']['wer_pct']}%)",
        f"**Snelste:** `{fastest['model']}` ({fastest['time_s']}s, "
        f"WER {fastest['wer']['wer_pct']}%)",
        "",
        "### WER Details per Model",
        "",
    ])

    for r in enrollment_results:
        w = r["wer"]
        lines.extend([
            f"#### {r['model']}",
            f"- WER: {w['wer_pct']}% ({w['errors']} fouten op {w['ref_words']} woorden)",
            f"- Substitutions: {w['substitutions']}, Insertions: {w['insertions']}, "
            f"Deletions: {w['deletions']}",
            f"- Taal: {r['language']} ({r['language_prob']})",
            f"- Preview: _{r['text_preview'][:150]}..._",
            "",
        ])

    if strategy_results:
        lines.extend([
            "---",
            "",
            "## 3. Strategiebespreking (57 min, multi-speaker)",
            "",
            "Vergelijking van medium en large-v3 op een echte vergadering met "
            "speaker diarization en identificatie.",
            "",
            "| Model | Transcriptie | Diarization | Totaal | RT Factor | WER | Sprekers |",
            "|-------|-------------|-------------|--------|-----------|-----|----------|",
        ])

        for r in strategy_results:
            w = r["wer"]
            speakers_str = ", ".join(
                f"{k} ({v})" for k, v in r["speaker_segment_counts"].items()
            )
            lines.append(
                f"| {r['model']} | {r['transcribe_time_s']}s | {r['diarize_time_s']}s | "
                f"{r['total_time_s']}s | {r['rt_factor']}x | "
                f"**{w['wer_pct']}%** | {speakers_str} |"
            )

        lines.extend([""])

        for r in strategy_results:
            lines.extend([
                f"### {r['model']} — Details",
                f"- Sprekers gedetecteerd: {r['speakers_detected']}",
                f"- Sprekers geidentificeerd: {r['speakers_identified']}",
                f"- Segmenten per spreker: {r['speaker_segment_counts']}",
                f"- WER: {r['wer']['wer_pct']}% "
                f"({r['wer']['errors']}/{r['wer']['ref_words']} fouten)",
                "",
            ])

    # Conclusion
    lines.extend([
        "---",
        "",
        "## 4. Conclusie: Transcripty vs. Plaud",
        "",
        "| Criterium | Plaud | Transcripty (best) |",
        "|-----------|-------|--------------------|",
        f"| Verwerkingstijd (105s audio) | ~60s (cloud) | "
        f"{best['time_s']}s (lokaal) |",
        f"| WER vs referentie | 0% (= referentie) | {best['wer']['wer_pct']}% |",
        f"| Beste model | — | `{best['model']}` |",
        f"| Snelste model | — | `{fastest['model']}` ({fastest['time_s']}s) |",
        f"| Speaker diarization | Ja (cloud) | Ja (pyannote, lokaal) |",
        f"| Speaker identification | Nee | Ja (SpeakerDB) |",
        f"| Privacy | Cloud | 100% lokaal |",
        f"| Kosten | Plaud abonnement | Gratis (open source) |",
        "",
    ])

    if strategy_results:
        strat_best = min(strategy_results, key=lambda x: x["wer"]["wer"])
        lines.extend([
            "### Strategiebespreking (57 min)",
            f"- Beste model: `{strat_best['model']}` "
            f"(WER {strat_best['wer']['wer_pct']}%)",
            f"- Totale verwerkingstijd: {strat_best['total_time_s']}s "
            f"({strat_best['rt_factor']}x realtime)",
            f"- Samuel Willems herkend: "
            f"{'Samuel Willems' in str(strat_best['speakers_identified'])}",
            "",
        ])

    report = "\n".join(lines)
    REPORT_FILE.write_text(report, encoding="utf-8")
    logger.info("\nReport saved to %s", REPORT_FILE)
    return report


def main():
    print(f"{'=' * 70}")
    print("TRANSCRIPTY BENCHMARK — Speaker Enrollment & Model Comparison")
    print(f"{'=' * 70}\n")

    # Step 1: Enroll
    enrollment_info = step1_enroll()
    if not enrollment_info:
        print("Enrollment failed. Aborting.")
        return

    # Step 2: Benchmark all models on enrollment audio
    enrollment_results = step2_benchmark_enrollment()

    # Step 3: Strategy benchmark (medium + large-v3)
    strategy_results = step3_strategy_benchmark()

    # Save raw results
    all_results = {
        "enrollment_info": enrollment_info,
        "enrollment_benchmark": [
            {k: v for k, v in r.items() if k != "full_text"}
            for r in enrollment_results
        ],
        "strategy_benchmark": strategy_results,
    }
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    logger.info("Raw results saved to %s", RESULTS_FILE)

    # Generate report
    report = generate_report(enrollment_info, enrollment_results, strategy_results)
    print(f"\n{'=' * 70}")
    print(report)


if __name__ == "__main__":
    main()
