"""Autoresearch Run 2 — Transcripty optimization.

Tests openstaande experimenten op basis van autoresearch.md agenda.
Baseline = optimale config uit Run 1 (vad + cond_prev=False + ngram=3).

Experiments (prioriteitsvolgorde):
  H7   — Max Suppression zonder temperature
  S5   — word_timestamps=False speed impact
  A4   — condition_on_previous_text=True + temperature=0.1
  T4   — temperature fallback tuple (0.0, 0.2, 0.4)
  T5   — beam_size=5 vs beam=1
  S1a  — cpu_threads=8
  S1b  — cpu_threads=10
  S1c  — cpu_threads=4
  M1   — multilingual=True
  M3   — language="nl" expliciete override (speed impact)
  C1   — hallucination_silence_threshold=2.0 (H7 variant)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.autoresearch_runner import run_suite  # noqa: E402

RESULTS_FILE = Path(__file__).parent / "autoresearch_results_run2.jsonl"

# Optimale config uit Run 1
BEST_CONFIG = dict(
    model_size="large-v3-turbo",
    compute_type="int8",
    beam_size=1,
    word_timestamps=True,
    temperature=0.0,
    vad_filter=True,
    condition_on_previous_text=False,
    repetition_penalty=1.0,
    no_repeat_ngram_size=3,
)

start_time = time.time()
print(f"\n{'╔' + '═' * 58 + '╗'}")
print(f"║  Autoresearch Run 2 — Transcripty                      ║")
print(f"║  Start: {time.strftime('%Y-%m-%d %H:%M')}                              ║")
print(f"{'╚' + '═' * 58 + '╝'}\n")


def log_result(exp_id, label, score, rtf, halluc, status, notes=""):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": 2,
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


def run_exp(exp_id, label, config, current_best_score, notes="", pre_hook=None, post_hook=None):
    """Run een experiment en beslis keep/discard."""
    print(f"\n🧪 Experiment {exp_id}: {label}")
    if pre_hook:
        pre_hook()
    r = run_suite(label, **config)
    if post_hook:
        post_hook()
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
    return (score if status == "KEEP" else current_best_score), score, status, rtf, halluc


results = []

# ─── BASELINE (warm — model al geladen) ──────────────────────────────────────
print("📊 Meting baseline (Run 1 optimale config)...")
b = run_suite("BASELINE_RUN2", **BEST_CONFIG)
BASELINE_SCORE = b["combined_score"]
BEST_SCORE = BASELINE_SCORE
print(f"\n  BASELINE: RTF={b['avg_rtf']:.3f}x | halluc={b['avg_halluc_rate']:.1%} | combined={BASELINE_SCORE:.4f}")
log_result("B2", "BASELINE_RUN2", BASELINE_SCORE, b["avg_rtf"], b["avg_halluc_rate"], "BASELINE")

# ─── H7: Max Suppression zonder temperature ───────────────────────────────────
cfg = dict(BEST_CONFIG,
    hallucination_silence_threshold=2.0,
    repetition_penalty=1.1,
)
BEST_SCORE, score, status, rtf, halluc = run_exp(
    "H7", "max_suppression_no_temp", cfg, BEST_SCORE,
    notes="vad+cond_prev=False+ngram=3+silence=2.0+penalty=1.1, geen temperature"
)
results.append(("H7", "max_suppression_no_temp", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg

# ─── S5: word_timestamps=False ────────────────────────────────────────────────
cfg = dict(BEST_CONFIG, word_timestamps=False)
BEST_SCORE, score, status, rtf, halluc = run_exp(
    "S5", "word_timestamps=False", cfg, BEST_SCORE,
    notes="Herverificatie: Echo Exp10 vond True 9% sneller — klopt nog met huidige config?"
)
results.append(("S5", "word_timestamps=False", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg

# ─── A4: condition_on_previous_text=True + temperature=0.1 ────────────────────
cfg = dict(BEST_CONFIG, condition_on_previous_text=True, temperature=0.1)
BEST_SCORE, score, status, rtf, halluc = run_exp(
    "A4", "cond_prev=True+temp=0.1", cfg, BEST_SCORE,
    notes="Context behouden + kleine randomness voor stabiliteit"
)
results.append(("A4", "cond_prev=True+temp=0.1", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg

# ─── T4: Temperature fallback tuple ───────────────────────────────────────────
cfg = dict(BEST_CONFIG, temperature=(0.0, 0.2, 0.4))
BEST_SCORE, score, status, rtf, halluc = run_exp(
    "T4", "temperature_fallback_tuple", cfg, BEST_SCORE,
    notes="faster-whisper fallback: 0.0 eerst, retry 0.2 bij lage confidence, dan 0.4"
)
results.append(("T4", "temperature_fallback_tuple", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg

# ─── T5: beam_size=5 ──────────────────────────────────────────────────────────
cfg = dict(BEST_CONFIG, beam_size=5)
BEST_SCORE, score, status, rtf, halluc = run_exp(
    "T5", "beam_size=5", cfg, BEST_SCORE,
    notes="Klassieke beam search vs greedy (beam=1)"
)
results.append(("T5", "beam_size=5", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg

# ─── S1: cpu_threads sweep ────────────────────────────────────────────────────
# cpu_threads wordt via configure() + model cache clear gezet
from transcripty.config import configure
from transcripty.transcribe import clear_model_cache

for threads, exp_id in [(8, "S1a"), (10, "S1b"), (4, "S1c"), (0, "S1d")]:
    label = f"cpu_threads={threads if threads > 0 else 'auto'}"
    cfg = dict(BEST_CONFIG)

    def _set_threads(t=threads):
        configure(cpu_threads=t)
        clear_model_cache()
        print(f"    [cpu_threads={t}, model cache cleared]")

    def _reset_threads():
        configure(cpu_threads=0)
        clear_model_cache()

    BEST_SCORE, score, status, rtf, halluc = run_exp(
        exp_id, label, cfg, BEST_SCORE,
        notes=f"M1 Max: 10 performance cores. cpu_threads={threads}",
        pre_hook=_set_threads,
        post_hook=_reset_threads,
    )
    results.append((exp_id, label, score, rtf, halluc, status))
    if status == "KEEP":
        BEST_CONFIG = dict(cfg)
        # Noteer de beste threads waarde
        BEST_CONFIG["_best_cpu_threads"] = threads

# Reset na threads experiment
configure(cpu_threads=0)
clear_model_cache()

# ─── M1: multilingual=True ────────────────────────────────────────────────────
cfg = dict(BEST_CONFIG)
cfg.pop("_best_cpu_threads", None)
cfg["multilingual"] = True
BEST_SCORE, score, status, rtf, halluc = run_exp(
    "M1", "multilingual=True", cfg, BEST_SCORE,
    notes="Per-chunk taaldetectie — verwacht: FR/NL opnames beter, NL-only: kleine overhead"
)
results.append(("M1", "multilingual=True", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg

# ─── M3: language="nl" expliciete override ────────────────────────────────────
cfg_m3 = dict(BEST_CONFIG)
cfg_m3.pop("_best_cpu_threads", None)
cfg_m3["language"] = "nl"
cfg_m3["multilingual"] = False
BEST_SCORE, score, status, rtf, halluc = run_exp(
    "M3", "language=nl+multilingual=False", cfg_m3, BEST_SCORE,
    notes="Forceert NL — verwacht sneller door skip auto-detect, maar FR/NL gaat fout"
)
results.append(("M3", "language=nl+multilingual=False", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg_m3

# ─── EINDRAPPORT ──────────────────────────────────────────────────────────────
elapsed_min = round((time.time() - start_time) / 60, 1)
n_kept = sum(1 for r in results if r[5] == "KEEP")
n_discarded = sum(1 for r in results if r[5] == "DISCARD")
improvement = round((BEST_SCORE - BASELINE_SCORE) / BASELINE_SCORE * 100, 1)

print(f"\n\n{'╔' + '═' * 58 + '╗'}")
print(f"║  AUTORESEARCH RUN 2 EINDRAPPORT                        ║")
print(f"{'╠' + '═' * 58 + '╣'}")
print(f"║  Gestopt:   {time.strftime('%Y-%m-%d %H:%M')}                              ║")
print(f"║  Duur:      {elapsed_min} minuten                                  ║")
print(f"║  Exp:       {len(results)} geprobeerd ({n_kept} behouden, {n_discarded} teruggedraaid)    ║")
print(f"║  Baseline:  {BASELINE_SCORE:.4f}                                    ║")
print(f"║  Best:      {BEST_SCORE:.4f} ({'+' if improvement >= 0 else ''}{improvement}%)                          ║")
print(f"{'╚' + '═' * 58 + '╝'}\n")

print(f"{'─' * 72}")
print(f"{'Exp':<6} {'Label':<32} {'RTF':>7} {'Halluc':>8} {'Score':>8} {'Status'}")
print(f"{'─' * 72}")
print(f"{'B2':<6} {'BASELINE_RUN2':<32} {b['avg_rtf']:>6.3f}x {b['avg_halluc_rate']:>7.1%} {BASELINE_SCORE:>8.4f}  BASELINE")
for (exp_id, label, score, rtf, halluc, status) in results:
    icon = "✅" if status == "KEEP" else "❌"
    print(f"{exp_id:<6} {label:<32} {rtf:>6.3f}x {halluc:>7.1%} {score:>8.4f}  {icon} {status}")

# Eindconfiguratie (verwijder interne keys)
final_config = {k: v for k, v in BEST_CONFIG.items() if not k.startswith("_")}

print(f"\n{'─' * 72}")
print(f"\n🏆 OPTIMALE CONFIG RUN 2 (combined_score={BEST_SCORE:.4f}, {'+' if improvement >= 0 else ''}{improvement}% vs baseline):")
for k, v in final_config.items():
    print(f"   {k}: {v}")

print(f"\n📄 Log: benchmarks/autoresearch_results_run2.jsonl")
