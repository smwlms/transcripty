"""Autonome autoresearch loop voor Transcripty.

Voert alle experimenten uit in volgorde van prioriteit.
Logt resultaten naar autoresearch_results.jsonl.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Voeg project root toe aan path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.autoresearch_runner import run_suite, BASELINE  # noqa: E402

RESULTS_FILE = Path(__file__).parent / "autoresearch_results.jsonl"

# Starttijd
start_time = time.time()
print(f"\n{'╔' + '═' * 58 + '╗'}")
print(f"║  Autoresearch Run — Transcripty                        ║")
print(f"║  Start: {time.strftime('%Y-%m-%d %H:%M')}                              ║")
print(f"{'╚' + '═' * 58 + '╝'}\n")


def log_result(exp_id, label, score, rtf, halluc, status, notes=""):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "exp": exp_id,
        "label": label,
        "score": score,
        "rtf": rtf,
        "halluc_rate": halluc,
        "status": status,
        "notes": notes,
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def run_exp(exp_id, label, config, current_best_score, notes=""):
    """Run een experiment en beslis keep/discard."""
    print(f"\n🧪 Experiment {exp_id}: {label}")
    r = run_suite(label, **config)
    score = r["combined_score"]
    rtf = r["avg_rtf"]
    halluc = r["avg_halluc_rate"]

    if score > current_best_score:
        status = "KEEP"
        delta = round((score - current_best_score) / current_best_score * 100, 1)
        print(f"  ✅ KEEP — score {score:.4f} > {current_best_score:.4f} (+{delta}%)")
    else:
        status = "DISCARD"
        delta = round((score - current_best_score) / current_best_score * 100, 1)
        print(f"  ❌ DISCARD — score {score:.4f} <= {current_best_score:.4f} ({delta}%)")

    log_result(exp_id, label, score, rtf, halluc, status, notes)
    return score if status == "KEEP" else current_best_score, score, status, rtf, halluc


results = []
consecutive_discards = 0

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────────────────────────────────────
print("📊 Meting baseline...")
b = run_suite("BASELINE", **BASELINE)
BASELINE_SCORE = b["combined_score"]
BEST_SCORE = BASELINE_SCORE
BEST_CONFIG = dict(BASELINE)
print(f"\n  BASELINE: RTF={b['avg_rtf']:.3f}x | halluc={b['avg_halluc_rate']:.1%} | combined={BASELINE_SCORE:.4f}")
log_result("B", "BASELINE", BASELINE_SCORE, b["avg_rtf"], b["avg_halluc_rate"], "BASELINE")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: H1 — vad_filter=True
# ─────────────────────────────────────────────────────────────────────────────
cfg = dict(BEST_CONFIG, vad_filter=True)
BEST_SCORE, score, status, rtf, halluc = run_exp("H1", "vad_filter=True", cfg, BEST_SCORE)
results.append(("H1", "vad_filter=True", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg
    consecutive_discards = 0
else:
    consecutive_discards += 1

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: T1 — temperature=0.2
# ─────────────────────────────────────────────────────────────────────────────
cfg = dict(BEST_CONFIG, temperature=0.2)
BEST_SCORE, score, status, rtf, halluc = run_exp("T1", "temperature=0.2", cfg, BEST_SCORE)
results.append(("T1", "temperature=0.2", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg
    consecutive_discards = 0
else:
    consecutive_discards += 1

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: H3 — hallucination_silence_threshold=2.0
# ─────────────────────────────────────────────────────────────────────────────
cfg = dict(BEST_CONFIG, hallucination_silence_threshold=2.0)
BEST_SCORE, score, status, rtf, halluc = run_exp("H3", "silence_threshold=2.0", cfg, BEST_SCORE)
results.append(("H3", "silence_threshold=2.0", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg
    consecutive_discards = 0
else:
    consecutive_discards += 1

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: H2 — condition_on_previous_text=False
# ─────────────────────────────────────────────────────────────────────────────
cfg = dict(BEST_CONFIG, condition_on_previous_text=False)
BEST_SCORE, score, status, rtf, halluc = run_exp("H2", "condition_previous=False", cfg, BEST_SCORE)
results.append(("H2", "condition_previous=False", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg
    consecutive_discards = 0
else:
    consecutive_discards += 1

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5: H4 — repetition_penalty=1.1
# ─────────────────────────────────────────────────────────────────────────────
cfg = dict(BEST_CONFIG, repetition_penalty=1.1)
BEST_SCORE, score, status, rtf, halluc = run_exp("H4", "repetition_penalty=1.1", cfg, BEST_SCORE)
results.append(("H4", "repetition_penalty=1.1", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg
    consecutive_discards = 0
else:
    consecutive_discards += 1

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 6: H5 — no_repeat_ngram_size=3
# ─────────────────────────────────────────────────────────────────────────────
cfg = dict(BEST_CONFIG, no_repeat_ngram_size=3)
BEST_SCORE, score, status, rtf, halluc = run_exp("H5", "no_repeat_ngram_size=3", cfg, BEST_SCORE)
results.append(("H5", "no_repeat_ngram_size=3", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg
    consecutive_discards = 0
else:
    consecutive_discards += 1

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 7: T2 — temperature=0.4
# ─────────────────────────────────────────────────────────────────────────────
# Temperature=0.2 was getest (T1). Nu 0.4 testen los van huidige config
cfg_t2 = dict(BEST_CONFIG)
if "temperature" not in cfg_t2 or cfg_t2.get("temperature") != 0.2:
    cfg_t2["temperature"] = 0.4
else:
    cfg_t2["temperature"] = 0.4
BEST_SCORE, score, status, rtf, halluc = run_exp("T2", "temperature=0.4", cfg_t2, BEST_SCORE)
results.append(("T2", "temperature=0.4", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg_t2
    consecutive_discards = 0
else:
    consecutive_discards += 1

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 8: A1a — repetition_penalty=1.05 (fijnere tuning)
# ─────────────────────────────────────────────────────────────────────────────
cfg = dict(BEST_CONFIG, repetition_penalty=1.05)
BEST_SCORE, score, status, rtf, halluc = run_exp("A1a", "repetition_penalty=1.05", cfg, BEST_SCORE)
results.append(("A1a", "repetition_penalty=1.05", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg
    consecutive_discards = 0
else:
    consecutive_discards += 1

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 9: H6 — Max Suppression Preset
# ─────────────────────────────────────────────────────────────────────────────
max_suppression = dict(
    BEST_CONFIG,
    vad_filter=True,
    condition_on_previous_text=False,
    hallucination_silence_threshold=2.0,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
    temperature=0.2,
)
BEST_SCORE, score, status, rtf, halluc = run_exp("H6", "MAX_SUPPRESSION", max_suppression, BEST_SCORE)
results.append(("H6", "MAX_SUPPRESSION", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = max_suppression
    consecutive_discards = 0
else:
    consecutive_discards += 1

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 10: T3 — temperature=0.2 + beam=1 (expliciete combinatie)
# ─────────────────────────────────────────────────────────────────────────────
cfg = dict(BEST_CONFIG, temperature=0.2, beam_size=1)
BEST_SCORE, score, status, rtf, halluc = run_exp("T3", "temp=0.2+beam=1", cfg, BEST_SCORE)
results.append(("T3", "temp=0.2+beam=1", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg
    consecutive_discards = 0
else:
    consecutive_discards += 1

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 11: A2 — no_repeat_ngram_size=2
# ─────────────────────────────────────────────────────────────────────────────
cfg = dict(BEST_CONFIG, no_repeat_ngram_size=2)
BEST_SCORE, score, status, rtf, halluc = run_exp("A2a", "no_repeat_ngram=2", cfg, BEST_SCORE)
results.append(("A2a", "no_repeat_ngram=2", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg
    consecutive_discards = 0
else:
    consecutive_discards += 1

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 12: S2 — VAD impact zonder andere wijzigingen (isolated)
# ─────────────────────────────────────────────────────────────────────────────
cfg = dict(BEST_CONFIG)
# Test met hogere beam voor kwaliteitscheck
cfg["beam_size"] = 3
BEST_SCORE, score, status, rtf, halluc = run_exp("A3", "beam_size=3", cfg, BEST_SCORE)
results.append(("A3", "beam_size=3", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg
    consecutive_discards = 0
else:
    consecutive_discards += 1

# ─────────────────────────────────────────────────────────────────────────────
# EINDRAPPORT
# ─────────────────────────────────────────────────────────────────────────────
elapsed_min = round((time.time() - start_time) / 60, 1)
n_kept = sum(1 for r in results if r[5] == "KEEP")
n_discarded = sum(1 for r in results if r[5] == "DISCARD")
improvement = round((BEST_SCORE - BASELINE_SCORE) / BASELINE_SCORE * 100, 1)

print(f"\n\n{'╔' + '═' * 58 + '╗'}")
print(f"║  AUTORESEARCH EINDRAPPORT                              ║")
print(f"{'╠' + '═' * 58 + '╣'}")
print(f"║  Gestopt:   {time.strftime('%Y-%m-%d %H:%M')}                              ║")
print(f"║  Duur:      {elapsed_min} minuten                                  ║")
print(f"║  Exp:       {len(results)} geprobeerd ({n_kept} behouden, {n_discarded} teruggedraaid)    ║")
print(f"║  Baseline:  {BASELINE_SCORE:.4f}                                    ║")
print(f"║  Best:      {BEST_SCORE:.4f} (+{improvement}%)                          ║")
print(f"{'╚' + '═' * 58 + '╝'}\n")

print("Resultaten per experiment:")
print(f"{'─' * 72}")
print(f"{'Exp':<6} {'Label':<28} {'RTF':>7} {'Halluc':>8} {'Score':>8} {'Status'}")
print(f"{'─' * 72}")
# Baseline
print(f"{'B':<6} {'BASELINE':<28} {b['avg_rtf']:>6.3f}x {b['avg_halluc_rate']:>7.1%} {BASELINE_SCORE:>8.4f}  BASELINE")
for (exp_id, label, score, rtf, halluc, status) in results:
    icon = "✅" if status == "KEEP" else "❌"
    print(f"{exp_id:<6} {label:<28} {rtf:>6.3f}x {halluc:>7.1%} {score:>8.4f}  {icon} {status}")

print(f"\n{'─' * 72}")
print(f"\n🏆 OPTIMALE CONFIG (combined_score={BEST_SCORE:.4f}, +{improvement}% vs baseline):")
for k, v in BEST_CONFIG.items():
    print(f"   {k}: {v}")

print(f"\n📄 Gedetailleerde log: benchmarks/autoresearch_results.jsonl")
