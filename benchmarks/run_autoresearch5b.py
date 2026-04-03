"""Autoresearch Run 5b — ECAPA tuning + Multilingual + Chunks + Presets.

Experiments:
  SI4b  — ECAPA weight tuning (try multiple weight combos + ECAPA-only)
  M2    — Multilingual FR/NL (corrected path)
  C1    — chunk_length=15
  C2    — chunk_length=60
  S4    — num_workers=2 parallel throughput
  P1-P3 — Finalize presets
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_FILE = Path(__file__).parent / "autoresearch_results_run5b.jsonl"
ECHO_ROOT = Path.home() / "Documents/Projecten/Plaude"

start_time = time.time()
print(f"\n{'╔' + '═' * 58 + '╗'}")
print(f"║  Autoresearch Run 5b — Tuning + Multilingual + Chunks    ║")
print(f"║  Start: {time.strftime('%Y-%m-%d %H:%M')}                              ║")
print(f"{'╚' + '═' * 58 + '╝'}\n")


def log_result(exp_id, label, metrics, status, notes=""):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": "5b", "exp": exp_id, "label": label,
        "metrics": metrics, "status": status, "notes": notes,
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: SI4b — ECAPA Weight Tuning
# ═══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PHASE 1: SI4b — ECAPA Weight & Threshold Tuning")
print("=" * 60)

from transcripty.diarize import diarize
from transcripty.ecapa import extract_ecapa_embeddings_for_segments
from transcripty.prosody import extract_prosodic_for_segments
from transcripty.speakers import SpeakerDB, _cosine_similarity

# Load enhanced DB
enhanced_db = SpeakerDB.load("benchmarks/speakers_enhanced_full.json")
pyannote_db = SpeakerDB.load("benchmarks/speakers_plaude.json")

# Test files
TEST_FILES = [
    {
        "path": ECHO_ROOT / "storage/2/e20d4d395b1d784ce6b6e4994c23887b.mp3",
        "label": "Roos gesprek", "duration": 191,
        "expected": ["Samuel", "Roos"],
    },
    {
        "path": ECHO_ROOT / "storage/2/3e156ded9f471f1297962d37dce3c845.mp3",
        "label": "audio NL", "duration": 313,
        "expected": ["Samuel"],
    },
]

# Diarize + extract features once for each file
file_features = []
for tf in TEST_FILES:
    if not tf["path"].exists():
        continue
    print(f"\n  Diarizing {tf['label']}...")
    t0 = time.time()
    diar = diarize(audio_path=str(tf["path"]))
    if not diar.embeddings:
        continue

    speaker_segs = {}
    for seg in diar.segments:
        speaker_segs.setdefault(seg.speaker, []).append((seg.start, seg.end))

    ecapa = extract_ecapa_embeddings_for_segments(str(tf["path"]), speaker_segs)
    prosody = extract_prosodic_for_segments(str(tf["path"]), speaker_segs)
    prosodic_vecs = {spk: f.as_vector() for spk, f in prosody.items()}
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s — {len(diar.embeddings)} speakers")

    file_features.append({
        "tf": tf, "diar": diar, "ecapa": ecapa, "prosodic_vecs": prosodic_vecs,
    })

# Test multiple weight/threshold combos
combos = [
    ("pyannote-only_t040", 0.40, (1.0, 0.0, 0.0)),
    ("pyannote-only_t035", 0.35, (1.0, 0.0, 0.0)),
    ("ecapa-only_t035", 0.35, (0.0, 1.0, 0.0)),
    ("ecapa-only_t025", 0.25, (0.0, 1.0, 0.0)),
    ("ecapa-only_t020", 0.20, (0.0, 1.0, 0.0)),
    ("enhanced_40-40-20_t040", 0.40, (0.4, 0.4, 0.2)),
    ("enhanced_40-40-20_t035", 0.35, (0.4, 0.4, 0.2)),
    ("enhanced_30-50-20_t035", 0.35, (0.3, 0.5, 0.2)),
    ("enhanced_20-60-20_t035", 0.35, (0.2, 0.6, 0.2)),
    ("enhanced_20-60-20_t030", 0.30, (0.2, 0.6, 0.2)),
    ("ecapa-heavy_10-80-10_t030", 0.30, (0.1, 0.8, 0.1)),
]

print(f"\n{'─' * 90}")
print(f"{'Combo':<32} {'Thresh':>6} {'Weights':>14}  ", end="")
for ff in file_features:
    print(f" {ff['tf']['label'][:12]:>12}", end="")
print(f"  {'Total':>6} {'False':>6}")
print(f"{'─' * 90}")

best_combo = None
best_score = -1  # correct - false_positives

for label, threshold, weights in combos:
    total_correct = 0
    total_false = 0
    file_results = []

    for ff in file_features:
        matches = enhanced_db.identify(
            ff["diar"],
            threshold=threshold,
            ecapa_embeddings=ff["ecapa"],
            prosodic_features=ff["prosodic_vecs"],
            weights=weights,
        )
        correct = sum(1 for n in matches.values() if n in ff["tf"]["expected"])
        false_pos = sum(1 for n in matches.values() if n not in ff["tf"]["expected"])
        total_correct += correct
        total_false += false_pos
        file_results.append(f"{correct}c/{false_pos}f/{len(matches)}t")

    score = total_correct - total_false
    w_str = f"{weights[0]:.1f}/{weights[1]:.1f}/{weights[2]:.1f}"
    print(f"{label:<32} {threshold:>6.2f} {w_str:>14}  ", end="")
    for fr in file_results:
        print(f" {fr:>12}", end="")
    icon = "🟢" if score > best_score else "  "
    print(f"  {total_correct:>6} {total_false:>6} {icon}")

    if score > best_score:
        best_score = score
        best_combo = (label, threshold, weights, total_correct, total_false)

if best_combo:
    print(f"\n  🏆 Best: {best_combo[0]} — {best_combo[3]} correct, {best_combo[4]} false positives")
    log_result("SI4b", f"best={best_combo[0]}", {
        "threshold": best_combo[1], "weights": list(best_combo[2]),
        "correct": best_combo[3], "false_positives": best_combo[4],
    }, "KEEP" if best_combo[3] > 2 else "INFO")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: M2 — Multilingual FR/NL (corrected path)
# ═══════════════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("PHASE 2: M2 — Multilingual FR/NL test")
print("=" * 60)

from transcripty import transcribe

BEST_CONFIG = dict(
    model_size="large-v3-turbo",
    compute_type="int8",
    beam_size=1,
    word_timestamps=True,
    temperature=(0.0, 0.2, 0.4),
    vad_filter=True,
    condition_on_previous_text=False,
    repetition_penalty=1.0,
    no_repeat_ngram_size=3,
)

# Corrected path: user 1, not user 2
ML_TEST = ECHO_ROOT / "storage/1/69f151e0d62f1e694e5e9d878a02badb.mp3"
ML_DUR = 49  # seconds

if ML_TEST.exists():
    configs = [
        ("A_nl_only", dict(language="nl", multilingual=False)),
        ("B_auto_detect", dict(language=None, multilingual=True)),
        ("C_nl_multilingual", dict(language="nl", multilingual=True)),
    ]

    ml_results = []
    for label, cfg in configs:
        print(f"\n  {label}...")
        t0 = time.time()
        r = transcribe(str(ML_TEST), **BEST_CONFIG, **cfg)
        dt = time.time() - t0
        rtf = round(dt / ML_DUR, 3)

        # Count language tags if available
        langs = {}
        for seg in r.segments:
            lang = getattr(seg, "language", None) or "?"
            langs[lang] = langs.get(lang, 0) + 1

        print(f"    {len(r.segments)} segs, RTF={rtf:.3f}x, lang={r.language}, per-seg: {langs}")

        # Show sample
        for seg in r.segments[:3]:
            lang_tag = f" [{seg.language}]" if seg.language else ""
            print(f"    {seg.start:.1f}-{seg.end:.1f}{lang_tag}: {seg.text[:70]}")

        ml_results.append({
            "label": label, "rtf": rtf, "segments": len(r.segments),
            "detected_lang": r.language, "lang_distribution": langs,
        })

    print(f"\n  RTF vergelijking:")
    for mr in ml_results:
        print(f"    {mr['label']:<20} RTF={mr['rtf']:.3f}x  segs={mr['segments']}  lang={mr['detected_lang']}  dist={mr['lang_distribution']}")

    log_result("M2", "multilingual_fr_nl", ml_results, "INFO")
else:
    print(f"  SKIP — {ML_TEST} not found")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: C1/C2 — Chunk Length (now supported via API)
# ═══════════════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("PHASE 3: C1/C2 — Chunk Length Experiments")
print("=" * 60)

from benchmarks.autoresearch_runner import run_suite, measure

# Use a single long file for chunk experiments (audio4, 585s)
LONG_FILE = ECHO_ROOT / "storage/2/0d1e65716cc93a4721e37e76683dacbb.mp3"
LONG_DUR = 585

if LONG_FILE.exists():
    # Warm up model first
    print("\n  Warming up model...")
    _ = transcribe(str(ML_TEST if ML_TEST.exists() else LONG_FILE),
                   **BEST_CONFIG, language="nl", multilingual=False)

    chunk_configs = [
        ("default_30", {}),
        ("C1_15", {"chunk_length": 15}),
        ("C2_60", {"chunk_length": 60}),
    ]

    c_results = []
    for label, extra in chunk_configs:
        print(f"\n  {label}...")
        cfg = dict(BEST_CONFIG, language="nl", multilingual=False, **extra)
        r = measure(LONG_FILE, LONG_DUR, **cfg)
        print(f"    RTF={r['rtf']:.3f}x  halluc={r['halluc_rate']:.1%}  segs={r['segments']}")
        c_results.append({"label": label, **r})

    baseline_c = c_results[0]
    for cr in c_results[1:]:
        exp_id = "C1" if "15" in cr["label"] else "C2"
        speed_score = 1 / cr["rtf"] if cr["rtf"] > 0 else 0
        quality_score = 1 - cr["halluc_rate"]
        score = speed_score * quality_score
        baseline_score = (1 / baseline_c["rtf"]) * (1 - baseline_c["halluc_rate"])

        if score > baseline_score:
            status = "KEEP"
            delta = round((score - baseline_score) / baseline_score * 100, 1)
            print(f"  ✅ {exp_id} KEEP — score {score:.2f} > {baseline_score:.2f} (+{delta}%)")
        else:
            status = "DISCARD"
            delta = round((score - baseline_score) / baseline_score * 100, 1)
            print(f"  ❌ {exp_id} DISCARD — score {score:.2f} <= {baseline_score:.2f} ({delta}%)")

        log_result(exp_id, cr["label"], {
            "rtf": cr["rtf"], "halluc_rate": cr["halluc_rate"],
            "segments": cr["segments"], "score": round(score, 2),
        }, status)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: P1/P2/P3 — Finalize Presets
# ═══════════════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("PHASE 4: P1/P2/P3 — Finalize Configuration Presets")
print("=" * 60)

presets = {
    "P1_speed": {
        "model_size": "large-v3-turbo",
        "compute_type": "int8",
        "beam_size": 1,
        "word_timestamps": True,
        "temperature": 0.0,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "no_repeat_ngram_size": 3,
        "language": "nl",
        "multilingual": False,
    },
    "P2_quality": {
        "model_size": "large-v3-turbo",
        "compute_type": "int8",
        "beam_size": 3,
        "word_timestamps": True,
        "temperature": 0.0,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "hallucination_silence_threshold": 2.0,
        "no_repeat_ngram_size": 3,
        "language": "nl",
        "multilingual": False,
    },
    "P3_balanced": {
        "model_size": "large-v3-turbo",
        "compute_type": "int8",
        "beam_size": 1,
        "word_timestamps": True,
        "temperature": (0.0, 0.2, 0.4),
        "vad_filter": True,
        "condition_on_previous_text": False,
        "no_repeat_ngram_size": 3,
        "language": "nl",
        "multilingual": False,
    },
}

# Warm run for each preset
print("\n  Running all 3 presets on test suite...")
preset_results = {}
for name, cfg in presets.items():
    print(f"\n  {name}:")
    r = run_suite(name, **cfg)
    preset_results[name] = r
    print(f"    RTF={r['avg_rtf']:.3f}x | halluc={r['avg_halluc_rate']:.1%} | score={r['combined_score']:.4f}")

# Summary table
print(f"\n{'─' * 60}")
print(f"{'Preset':<16} {'RTF':>8} {'Halluc':>8} {'Score':>8}")
print(f"{'─' * 60}")
for name, r in preset_results.items():
    print(f"{name:<16} {r['avg_rtf']:>7.3f}x {r['avg_halluc_rate']:>7.1%} {r['combined_score']:>8.4f}")

log_result("P1", "speed_preset", preset_results.get("P1_speed", {}), "DEFINED")
log_result("P2", "quality_preset", preset_results.get("P2_quality", {}), "DEFINED")
log_result("P3", "balanced_preset", preset_results.get("P3_balanced", {}), "DEFINED")


# ═══════════════════════════════════════════════════════════════════════
# EINDRAPPORT
# ═══════════════════════════════════════════════════════════════════════

elapsed_min = round((time.time() - start_time) / 60, 1)

print(f"\n\n{'╔' + '═' * 58 + '╗'}")
print(f"║  AUTORESEARCH RUN 5b EINDRAPPORT                       ║")
print(f"{'╠' + '═' * 58 + '╣'}")
print(f"║  Duur:      {elapsed_min} minuten                                  ║")
print(f"{'╚' + '═' * 58 + '╝'}\n")

print(f"📄 Log: {RESULTS_FILE}")
print(f"⏱️  Totale duur: {elapsed_min} minuten")
