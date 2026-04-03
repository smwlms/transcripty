"""Vergelijking: large-v3-turbo vs large-v3

Test op de 881a40bd opname (61 min, NL/FR/EN, 4 sprekers).
Runs ALLEEN transcriptie (geen diarization) voor snelheidsvergelijking.
Toont ook kwaliteitsverschil via eerste 30 segmenten.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transcripty import transcribe
from transcripty.transcribe import clear_model_cache

AUDIO = Path.home() / "Documents/Projecten/Plaude/storage/1/881a40bd2e0586268640f9d94efd56f9.mp3"
DURATION = 3706.8  # seconden (61.8 min)

COMMON = dict(
    vad_filter=True,
    condition_on_previous_text=False,
    no_repeat_ngram_size=3,
    beam_size=1,
    word_timestamps=True,
    multilingual=True,   # NL/FR/EN detectie per segment
    language=None,       # auto-detect
    temperature=(0.0, 0.2, 0.4),
)

def run(label, **kwargs):
    print(f"\n{'═'*60}")
    print(f"Model: {label}")
    print(f"Params: {kwargs}")
    print(f"{'═'*60}")
    clear_model_cache()
    t0 = time.time()
    result = transcribe(str(AUDIO), **kwargs)
    elapsed = time.time() - t0
    rtf = elapsed / DURATION
    segs = result.segments
    n_halluc = sum(1 for s in segs if len(s.text.strip()) < 3)
    langs = {}
    for s in segs:
        if s.language:
            langs[s.language] = langs.get(s.language, 0) + 1
    total = sum(langs.values())
    lang_pct = {k: round(v*100/total) for k, v in sorted(langs.items(), key=lambda x: -x[1])} if total else {}

    print(f"\n  ⏱  Tijd: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  🚀 RTF:  {rtf:.3f}x realtime")
    print(f"  📝 Segmenten: {len(segs)}")
    print(f"  🌍 Talen: {lang_pct}")
    print(f"  ✅ Hallucinations: {n_halluc}")

    print(f"\n  --- Eerste 20 segmenten ---")
    for s in segs[:20]:
        lang_tag = f"[{s.language}] " if s.language else ""
        print(f"  {s.start:6.1f}s  {lang_tag}{s.text.strip()}")

    return {
        "label": label,
        "elapsed_s": round(elapsed, 1),
        "rtf": round(rtf, 3),
        "segments": len(segs),
        "languages": lang_pct,
        "hallucinations": n_halluc,
        "text_sample": " ".join(s.text for s in segs[:50]),
    }

print(f"\n{'╔'+'═'*58+'╗'}")
print(f"║  Turbo vs Large-v3 vergelijking                        ║")
print(f"║  Bestand: 881a40bd (61.8 min, NL/FR/EN, 4 sprekers)   ║")
print(f"{'╚'+'═'*58+'╝'}")

# ─── Config A: large-v3-turbo ─────────────────────────────────────────────────
result_turbo = run("large-v3-turbo", model_size="large-v3-turbo", compute_type="int8", **COMMON)

# ─── Config B: large-v3 (volledig) ────────────────────────────────────────────
result_large = run("large-v3", model_size="large-v3", compute_type="int8", **COMMON)

# ─── Vergelijking ─────────────────────────────────────────────────────────────
print(f"\n\n{'╔'+'═'*58+'╗'}")
print(f"║  VERGELIJKING                                          ║")
print(f"{'╠'+'═'*58+'╣'}")
speedup = round(result_large["rtf"] / result_turbo["rtf"], 1)
print(f"║  Turbo RTF:  {result_turbo['rtf']:.3f}x  ({result_turbo['elapsed_s']/60:.1f} min voor 61.8 min audio) ║")
print(f"║  Large RTF:  {result_large['rtf']:.3f}x  ({result_large['elapsed_s']/60:.1f} min voor 61.8 min audio) ║")
print(f"║  Turbo is {speedup}x sneller dan large-v3                 ║")
print(f"║                                                        ║")
print(f"║  Turbo segmenten: {result_turbo['segments']:<6}  Large: {result_large['segments']:<6}              ║")
print(f"║  Turbo talen: {result_turbo['languages']}  ║")
print(f"║  Large talen: {result_large['languages']}  ║")
print(f"{'╚'+'═'*58+'╝'}")

import json
out = Path(__file__).parent / "compare_turbo_large_result.json"
with open(out, "w") as f:
    json.dump({"turbo": result_turbo, "large": result_large}, f, indent=2, ensure_ascii=False)
print(f"\n📄 Resultaten: {out}")
