"""Autoresearch Run 4 — Speaker Identification + Vocabulary experiments.

Open experiments from agenda:
  SI1  — threshold 0.50 → 0.40
  SI3  — threshold 0.50 → 0.45
  SI2  — threshold 0.50 → 0.35
  V1   — Vastgoed vocabulary (NL)
  V2   — Speaker namen als vocabulary
  V3   — V1 + V2 combined

Metrics:
  SI experiments: speaker_accuracy (matched / total_speakers_in_audio)
  V experiments:  combined_score (1/RTF * (1 - halluc_rate))
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_FILE = Path(__file__).parent / "autoresearch_results_run4.jsonl"
ECHO_ROOT = Path.home() / "Documents/Projecten/Plaude"
TRANSCRIPTY_ROOT = Path(__file__).parent.parent

# Speaker DB path
SPEAKER_DB = TRANSCRIPTY_ROOT / "speakers.json"

start_time = time.time()
print(f"\n{'╔' + '═' * 58 + '╗'}")
print(f"║  Autoresearch Run 4 — Speaker ID + Vocabulary            ║")
print(f"║  Start: {time.strftime('%Y-%m-%d %H:%M')}                              ║")
print(f"{'╚' + '═' * 58 + '╝'}\n")


def log_result(exp_id, label, metrics, status, notes=""):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": 4,
        "exp": exp_id,
        "label": label,
        "metrics": metrics,
        "status": status,
        "notes": notes,
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


# ═══════════════════════════════════════════════════════════════════════
# PART A: Speaker Identification Threshold Experiments (SI1, SI2, SI3)
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART A: Speaker Identification Threshold Sweep")
print("=" * 60)

# Test file: Roos gesprek (53) — 191s, 3 speakers (Samuel, Roos, +1)
SI_TEST_FILE = ECHO_ROOT / "storage/2/e20d4d395b1d784ce6b6e4994c23887b.mp3"
SI_TEST_DURATION = 191

if not SI_TEST_FILE.exists():
    print(f"❌ Test bestand niet gevonden: {SI_TEST_FILE}")
    print("   Skip SI experimenten.")
    si_results = []
else:
    print(f"\n📁 Test file: {SI_TEST_FILE.name} ({SI_TEST_DURATION}s)")
    print("🔊 Stap 1: Diarization (eenmalig)...")

    from transcripty.diarize import diarize
    from transcripty.speakers import SpeakerDB

    diarize_start = time.time()
    diarization_result = diarize(
        audio_path=str(SI_TEST_FILE),
        hf_token=None,  # uses HF_TOKEN env var
    )
    diarize_elapsed = round(time.time() - diarize_start, 1)

    n_speakers_detected = len(diarization_result.embeddings) if diarization_result.embeddings else 0
    n_segments = len(diarization_result.segments)
    print(f"   Diarization klaar in {diarize_elapsed}s — {n_speakers_detected} speakers, {n_segments} segmenten")

    # Load speaker DB
    print("\n🔊 Stap 2: Speaker DB laden...")
    speaker_db = SpeakerDB.load(str(SPEAKER_DB))
    print(f"   {len(speaker_db)} enrolled speakers: {speaker_db.names}")

    # Test multiple thresholds
    thresholds = [0.50, 0.45, 0.40, 0.35, 0.30]
    si_results = []

    print("\n🔊 Stap 3: Threshold sweep...")
    print(f"\n{'─' * 72}")
    print(f"{'Threshold':>10} {'Matched':>8} {'UNKNOWN':>8} {'Match%':>8} {'Details'}")
    print(f"{'─' * 72}")

    # Also collect all pairwise scores for analysis
    print("\n📊 Pairwise cosine similarity scores:")
    if diarization_result.embeddings:
        from transcripty.speakers import _cosine_similarity

        print(f"\n{'':>14}", end="")
        for name in speaker_db.names:
            print(f"{name:>12}", end="")
        print()

        for speaker_label, speaker_emb in diarization_result.embeddings.items():
            print(f"{speaker_label:>14}", end="")
            for name in speaker_db.names:
                profile = speaker_db.profiles[name]
                score = _cosine_similarity(speaker_emb, profile.embedding)
                marker = " *" if score >= 0.35 else ""
                print(f"{score:>10.3f}{marker}", end="")
            print()

    print(f"\n{'─' * 72}")
    print(f"{'Threshold':>10} {'Matched':>8} {'UNKNOWN':>8} {'Match%':>8} {'Details'}")
    print(f"{'─' * 72}")

    for threshold in thresholds:
        matches = speaker_db.identify(diarization_result, threshold=threshold)
        n_matched = len(matches)
        n_unknown = n_speakers_detected - n_matched

        match_pct = round(n_matched / n_speakers_detected * 100, 1) if n_speakers_detected > 0 else 0

        details = ", ".join(f"{lbl}→{name}" for lbl, name in matches.items())
        if not details:
            details = "(geen matches)"

        print(f"{threshold:>10.2f} {n_matched:>8} {n_unknown:>8} {match_pct:>7.1f}% {details}")

        si_results.append({
            "threshold": threshold,
            "matched": n_matched,
            "unknown": n_unknown,
            "total_speakers": n_speakers_detected,
            "match_pct": match_pct,
            "matches": matches,
        })

    # Determine best threshold: maximize matched without false positives
    # Baseline is 0.50
    baseline_si = si_results[0]  # threshold=0.50
    print(f"\n📊 Baseline (threshold=0.50): {baseline_si['matched']}/{baseline_si['total_speakers']} matched ({baseline_si['match_pct']}%)")

    for sr in si_results[1:]:
        threshold = sr["threshold"]
        exp_map = {0.45: "SI3", 0.40: "SI1", 0.35: "SI2", 0.30: "SI_extra"}
        exp_id = exp_map.get(threshold, f"SI_{threshold}")

        if sr["matched"] > baseline_si["matched"]:
            status = "KEEP"
            delta = sr["matched"] - baseline_si["matched"]
            print(f"\n  ✅ {exp_id} (threshold={threshold}): KEEP — +{delta} matches ({sr['match_pct']}%)")
        else:
            status = "DISCARD"
            print(f"\n  ❌ {exp_id} (threshold={threshold}): DISCARD — same or fewer matches")

        log_result(exp_id, f"threshold={threshold}", sr, status,
                   notes=f"matches: {sr['matches']}")


# ═══════════════════════════════════════════════════════════════════════
# PART B: Vocabulary Experiments (V1, V2, V3)
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART B: Vocabulary / Initial Prompt Experiments")
print("=" * 60)

from benchmarks.autoresearch_runner import run_suite

# Optimale config uit Run 2 (zonder vocabulary)
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
    language="nl",
    multilingual=False,
)

# ─── BASELINE (Run 2 best config, no prompt) ────────────────────────
print("\n📊 Baseline meting (geen vocabulary)...")
b = run_suite("BASELINE_V", **BEST_CONFIG)
BASELINE_SCORE = b["combined_score"]
BEST_SCORE = BASELINE_SCORE
BASELINE_RTF = b["avg_rtf"]
print(f"\n  BASELINE: RTF={b['avg_rtf']:.3f}x | halluc={b['avg_halluc_rate']:.1%} | combined={BASELINE_SCORE:.4f}")
log_result("B4", "BASELINE_V", {
    "score": BASELINE_SCORE,
    "rtf": b["avg_rtf"],
    "halluc_rate": b["avg_halluc_rate"],
}, "BASELINE")

# ─── V1: Vastgoed vocabulary ────────────────────────────────────────
VASTGOED_VOCAB = [
    "compromis", "akte", "EPC", "kadastraal inkomen", "onroerende voorheffing",
    "erfdienstbaarheid", "riolering", "stedenbouwkundige vergunning", "notaris",
    "immo", "pand", "bod", "opschortende voorwaarden", "hypotheek", "lening",
    "makelaardij", "commissie", "exclusiviteit", "overnamebeding",
    "Whise", "Plaud", "Claes & Willems",
]

v1_prompt = ", ".join(VASTGOED_VOCAB)
cfg_v1 = dict(BEST_CONFIG, prompt=v1_prompt)
print(f"\n🧪 V1: Vastgoed vocabulary ({len(VASTGOED_VOCAB)} woorden)")
print(f"   Prompt: {v1_prompt[:80]}...")
r_v1 = run_suite("V1_vastgoed_vocab", **cfg_v1)
v1_score = r_v1["combined_score"]
v1_rtf = r_v1["avg_rtf"]
v1_halluc = r_v1["avg_halluc_rate"]

if v1_score > BEST_SCORE:
    v1_status = "KEEP"
    delta = round((v1_score - BEST_SCORE) / BEST_SCORE * 100, 1)
    print(f"  ✅ V1 KEEP — score {v1_score:.4f} > {BEST_SCORE:.4f} (+{delta}%)")
    BEST_SCORE = v1_score
    BEST_CONFIG_V = cfg_v1
else:
    v1_status = "DISCARD"
    delta = round((v1_score - BEST_SCORE) / BEST_SCORE * 100, 1)
    print(f"  ❌ V1 DISCARD — score {v1_score:.4f} <= {BEST_SCORE:.4f} ({delta}%)")
    BEST_CONFIG_V = BEST_CONFIG

log_result("V1", "vastgoed_vocab", {
    "score": v1_score, "rtf": v1_rtf, "halluc_rate": v1_halluc,
    "vocab_size": len(VASTGOED_VOCAB),
}, v1_status)

# ─── V2: Speaker namen als prompt ───────────────────────────────────
SPEAKER_NAMES = ["Samuel", "Alexander", "Roos", "Andries", "Evi", "Nico", "Deborah"]
v2_prompt = ", ".join(SPEAKER_NAMES)
cfg_v2 = dict(BEST_CONFIG, prompt=v2_prompt)
print(f"\n🧪 V2: Speaker namen als prompt ({len(SPEAKER_NAMES)} namen)")
print(f"   Prompt: {v2_prompt}")
r_v2 = run_suite("V2_speaker_names", **cfg_v2)
v2_score = r_v2["combined_score"]
v2_rtf = r_v2["avg_rtf"]
v2_halluc = r_v2["avg_halluc_rate"]

if v2_score > BEST_SCORE:
    v2_status = "KEEP"
    delta = round((v2_score - BEST_SCORE) / BEST_SCORE * 100, 1)
    print(f"  ✅ V2 KEEP — score {v2_score:.4f} > {BEST_SCORE:.4f} (+{delta}%)")
    BEST_SCORE = v2_score
else:
    v2_status = "DISCARD"
    delta = round((v2_score - BEST_SCORE) / BEST_SCORE * 100, 1)
    print(f"  ❌ V2 DISCARD — score {v2_score:.4f} <= {BEST_SCORE:.4f} ({delta}%)")

log_result("V2", "speaker_names", {
    "score": v2_score, "rtf": v2_rtf, "halluc_rate": v2_halluc,
    "names": SPEAKER_NAMES,
}, v2_status)

# ─── V3: Combinatie vastgoed + namen ────────────────────────────────
v3_vocab = VASTGOED_VOCAB + SPEAKER_NAMES
v3_prompt = ", ".join(v3_vocab)
cfg_v3 = dict(BEST_CONFIG, prompt=v3_prompt)
print(f"\n🧪 V3: Vastgoed + namen ({len(v3_vocab)} woorden)")
print(f"   Prompt: {v3_prompt[:80]}...")
r_v3 = run_suite("V3_combined_vocab", **cfg_v3)
v3_score = r_v3["combined_score"]
v3_rtf = r_v3["avg_rtf"]
v3_halluc = r_v3["avg_halluc_rate"]

if v3_score > BEST_SCORE:
    v3_status = "KEEP"
    delta = round((v3_score - BEST_SCORE) / BEST_SCORE * 100, 1)
    print(f"  ✅ V3 KEEP — score {v3_score:.4f} > {BEST_SCORE:.4f} (+{delta}%)")
    BEST_SCORE = v3_score
else:
    v3_status = "DISCARD"
    delta = round((v3_score - BEST_SCORE) / BEST_SCORE * 100, 1)
    print(f"  ❌ V3 DISCARD — score {v3_score:.4f} <= {BEST_SCORE:.4f} ({delta}%)")

log_result("V3", "combined_vocab", {
    "score": v3_score, "rtf": v3_rtf, "halluc_rate": v3_halluc,
    "vocab_size": len(v3_vocab),
}, v3_status)


# ═══════════════════════════════════════════════════════════════════════
# EINDRAPPORT
# ═══════════════════════════════════════════════════════════════════════

elapsed_min = round((time.time() - start_time) / 60, 1)

print(f"\n\n{'╔' + '═' * 58 + '╗'}")
print(f"║  AUTORESEARCH RUN 4 EINDRAPPORT                        ║")
print(f"{'╠' + '═' * 58 + '╣'}")
print(f"║  Gestopt:   {time.strftime('%Y-%m-%d %H:%M')}                              ║")
print(f"║  Duur:      {elapsed_min} minuten                                  ║")
print(f"{'╚' + '═' * 58 + '╝'}\n")

print("PART A — Speaker Identification Threshold Sweep:")
print(f"{'─' * 72}")
if si_results:
    for sr in si_results:
        icon = "🟢" if sr["matched"] > baseline_si["matched"] else "⚪"
        print(f"  {icon} threshold={sr['threshold']:.2f}  matched={sr['matched']}/{sr['total_speakers']}  ({sr['match_pct']:.1f}%)")
else:
    print("  (geen resultaten — test bestand niet gevonden)")

print(f"\nPART B — Vocabulary Experiments:")
print(f"{'─' * 72}")
v_baseline_info = f"RTF={BASELINE_RTF:.3f}x | score={BASELINE_SCORE:.4f}"
print(f"  BASELINE: {v_baseline_info}")
for exp_id, label, score, rtf, halluc, status in [
    ("V1", "vastgoed_vocab", v1_score, v1_rtf, v1_halluc, v1_status),
    ("V2", "speaker_names", v2_score, v2_rtf, v2_halluc, v2_status),
    ("V3", "combined_vocab", v3_score, v3_rtf, v3_halluc, v3_status),
]:
    icon = "✅" if status == "KEEP" else "❌"
    delta = round((score - BASELINE_SCORE) / BASELINE_SCORE * 100, 1) if BASELINE_SCORE > 0 else 0
    print(f"  {icon} {exp_id}: {label:<20} RTF={rtf:.3f}x  halluc={halluc:.1%}  score={score:.4f}  ({'+' if delta >= 0 else ''}{delta}%)")

print(f"\n📄 Log: benchmarks/autoresearch_results_run4.jsonl")
print(f"⏱️  Totale duur: {elapsed_min} minuten")
