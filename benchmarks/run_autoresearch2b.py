"""Autoresearch Run 2b — vervolg na crash op S1d.

Baseline = beste config na Run 2 (T4 + S1c):
  temperature=(0.0, 0.2, 0.4), cpu_threads=4

Nog te testen: S1d (auto threads vergelijking), M1, M3
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.autoresearch_runner import run_suite
from transcripty.config import configure
from transcripty.transcribe import clear_model_cache

RESULTS_FILE = Path(__file__).parent / "autoresearch_results_run2.jsonl"

# Beste config na Run 2 (T4 + S1c)
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
BEST_SCORE = 4.0878  # S1c resultaat

start_time = time.time()
print(f"\n{'╔' + '═' * 58 + '╗'}")
print(f"║  Autoresearch Run 2b — vervolg                         ║")
print(f"║  Start: {time.strftime('%Y-%m-%d %H:%M')}                              ║")
print(f"║  Baseline (T4+S1c): score={BEST_SCORE}, RTF=0.245x     ║")
print(f"{'╚' + '═' * 58 + '╝'}\n")

# Zet cpu_threads=4 als startpunt
configure(cpu_threads=4)
clear_model_cache()


def log_result(exp_id, label, score, rtf, halluc, status, notes=""):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": "2b",
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

# ─── S1d: cpu_threads=auto (vergelijking met 4) ───────────────────────────────
def _set_auto():
    configure(cpu_threads=0)
    clear_model_cache()
    print("    [cpu_threads=auto, model cache cleared]")

def _set_4():
    configure(cpu_threads=4)
    clear_model_cache()
    print("    [cpu_threads=4, model cache restored]")

BEST_SCORE, score, status, rtf, halluc = run_exp(
    "S1d", "cpu_threads=auto", BEST_CONFIG, BEST_SCORE,
    notes="Controle: is auto beter dan 4 op M1 Max?",
    pre_hook=_set_auto,
    post_hook=_set_4,
)
results.append(("S1d", "cpu_threads=auto", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = dict(BEST_CONFIG)  # auto = cpu_threads=0 is default

# ─── M1: multilingual=True ────────────────────────────────────────────────────
cfg = dict(BEST_CONFIG, multilingual=True)
BEST_SCORE, score, status, rtf, halluc = run_exp(
    "M1", "multilingual=True", cfg, BEST_SCORE,
    notes="Per-chunk taaldetectie via lingua — FR/NL opnames beter"
)
results.append(("M1", "multilingual=True", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg

# ─── M3: language="nl" expliciete override ────────────────────────────────────
cfg_m3 = dict(BEST_CONFIG, language="nl", multilingual=False)
BEST_SCORE, score, status, rtf, halluc = run_exp(
    "M3", "language=nl+multilingual=False", cfg_m3, BEST_SCORE,
    notes="Forceert NL — sneller door skip auto-detect, maar FR gaat fout"
)
results.append(("M3", "language=nl+multilingual=False", score, rtf, halluc, status))
if status == "KEEP":
    BEST_CONFIG = cfg_m3

# ─── EINDRAPPORT ──────────────────────────────────────────────────────────────
elapsed_min = round((time.time() - start_time) / 60, 1)
n_kept = sum(1 for r in results if r[5] == "KEEP")
n_discarded = sum(1 for r in results if r[5] == "DISCARD")
BASELINE_2B = 4.0878
improvement = round((BEST_SCORE - BASELINE_2B) / BASELINE_2B * 100, 1)

print(f"\n\n{'╔' + '═' * 58 + '╗'}")
print(f"║  AUTORESEARCH RUN 2b EINDRAPPORT                       ║")
print(f"{'╠' + '═' * 58 + '╣'}")
print(f"║  Gestopt:   {time.strftime('%Y-%m-%d %H:%M')}                              ║")
print(f"║  Duur:      {elapsed_min} min                                      ║")
print(f"║  Exp:       {len(results)} geprobeerd ({n_kept} KEEP, {n_discarded} DISCARD)           ║")
print(f"║  Baseline:  {BASELINE_2B:.4f}                                    ║")
print(f"║  Best:      {BEST_SCORE:.4f} ({'+' if improvement >= 0 else ''}{improvement}%)                          ║")
print(f"{'╚' + '═' * 58 + '╝'}\n")

print(f"{'─' * 72}")
print(f"{'Exp':<6} {'Label':<32} {'RTF':>7} {'Halluc':>8} {'Score':>8} {'Status'}")
print(f"{'─' * 72}")
for (exp_id, label, score, rtf, halluc, status) in results:
    icon = "✅" if status == "KEEP" else "❌"
    print(f"{exp_id:<6} {label:<32} {rtf:>6.3f}x {halluc:>7.1%} {score:>8.4f}  {icon} {status}")

print(f"\n🏆 FINALE OPTIMALE CONFIG (Run 1+2+2b):")
for k, v in BEST_CONFIG.items():
    print(f"   {k}: {v}")
print(f"\n📄 Log: benchmarks/autoresearch_results_run2.jsonl")
