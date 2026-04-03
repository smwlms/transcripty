import os
"""Autoresearch Run 3 — Multilingual Per-Word Detection experiments.

Test bestand: 49s audio (Samuel NL + Evi FR), 2 sprekers.
Metric: RTF + multilingual_accuracy (% correct taal-tags).

Ground truth (handmatig bepaald):
- 0-18s: NL (Samuel + Evi)
- 18-49s: FR (Evi)
- "d'accord" @16s = FR (maar in NL context, mag als NL tellen)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydub import AudioSegment
from bisect import bisect_right
from transcripty.config import configure
from transcripty.transcribe import _get_model, clear_model_cache
from transcripty.diarize import diarize
from transcripty.device import detect_device
from transcripty.speakers import SpeakerDB
from transcripty.models import Word, LabeledSegment

AUDIO = Path.home() / "Documents/Projecten/Plaude/storage/1/69f151e0d62f1e694e5e9d878a02badb.mp3"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SAMPLE_RATE = 16000

# Ground truth: time ranges and expected language
GROUND_TRUTH = [
    (0.0, 18.0, "nl"),
    (18.0, 49.0, "fr"),
]


def load_speaker_db() -> SpeakerDB:
    from sqlalchemy import create_engine, text as sqlt
    engine = create_engine("sqlite:///" + str(Path.home() / "Documents/Projecten/Plaude/echo.db"))
    with engine.connect() as conn:
        speakers = conn.execute(sqlt(
            "SELECT name, voiceprint FROM speakers WHERE is_active=1 AND voiceprint IS NOT NULL"
        )).fetchall()
    db = SpeakerDB()
    for name, vp in speakers:
        db.enroll_from_embedding(name, json.loads(vp.decode("utf-8")))
    return db


def load_audio_numpy() -> np.ndarray:
    audio = AudioSegment.from_file(str(AUDIO))
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    return np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0


def get_expected_lang(t: float) -> str:
    for start, end, lang in GROUND_TRUTH:
        if start <= t < end:
            return lang
    return "nl"


def measure_accuracy(segments: list) -> dict:
    """Measure multilingual accuracy against ground truth."""
    correct = 0
    total = 0
    errors = []
    for seg in segments:
        if not seg.get("text", "").strip():
            continue
        mid_time = (seg["start"] + seg["end"]) / 2
        expected = get_expected_lang(mid_time)
        actual = seg.get("language", "?")
        total += 1
        if actual == expected:
            correct += 1
        else:
            errors.append(f"  {seg['start']:.1f}s: expected={expected} got={actual} '{seg['text'][:50]}'")
    accuracy = correct / total if total else 0
    return {"accuracy": round(accuracy * 100, 1), "correct": correct, "total": total, "errors": errors}


def find_speaker(start, end, diar_segments, diar_starts, names):
    overlaps = {}
    idx = bisect_right(diar_starts, start)
    for i in range(max(0, idx - 1), len(diar_segments)):
        d = diar_segments[i]
        if d.start >= end:
            break
        overlap = min(end, d.end) - max(start, d.start)
        if overlap > 0:
            overlaps[d.speaker] = overlaps.get(d.speaker, 0) + overlap
    if not overlaps:
        return "?"
    label = max(overlaps, key=lambda k: overlaps[k])
    return (names or {}).get(label, label)


def run_experiment(name: str, window_size: float, samples: np.ndarray,
                   model, diar, names, duration: float) -> dict:
    """Run a windowed transcription experiment."""
    print(f"\n  🧪 {name} (window={window_size}s)")
    t0 = time.time()

    diar_starts = [d.start for d in diar.segments]
    n_windows = int(np.ceil(duration / window_size))
    segments = []

    for wi in range(n_windows):
        win_start = wi * window_size
        win_end = min((wi + 1) * window_size, duration)
        s0 = int(win_start * SAMPLE_RATE)
        s1 = min(int(win_end * SAMPLE_RATE), len(samples))
        chunk = samples[s0:s1]

        if len(chunk) < SAMPLE_RATE * 0.3:
            continue

        segs_gen, info = model.transcribe(chunk, language=None, word_timestamps=True,
                                           beam_size=1, condition_on_previous_text=False,
                                           no_repeat_ngram_size=3, vad_filter=False)

        for seg in segs_gen:
            text = seg.text.strip()
            if text:
                abs_start = seg.start + win_start
                abs_end = seg.end + win_start
                spk = find_speaker(abs_start, abs_end, diar.segments, diar_starts, names)
                segments.append({
                    "start": abs_start, "end": abs_end,
                    "speaker": spk, "language": info.language, "text": text,
                })

    elapsed = time.time() - t0
    rtf = elapsed / duration
    acc = measure_accuracy(segments)

    print(f"     ⏱  {elapsed:.1f}s (RTF {rtf:.3f}x)")
    print(f"     📊 Accuracy: {acc['accuracy']}% ({acc['correct']}/{acc['total']})")
    if acc["errors"]:
        for e in acc["errors"][:5]:
            print(f"     ❌ {e}")

    return {
        "name": name, "window_size": window_size,
        "elapsed_s": round(elapsed, 1), "rtf": round(rtf, 3),
        "n_segments": len(segments), "n_windows": n_windows,
        "accuracy": acc["accuracy"], "correct": acc["correct"], "total": acc["total"],
        "segments": segments,
    }


def run_hybrid(name: str, samples: np.ndarray, model, diar, names, duration: float) -> dict:
    """ML7: Hybrid — transcribe whole as NL, detect lang per window, retranscribe non-NL."""
    print(f"\n  🧪 {name} (hybrid: NL whole + retranscribe FR/EN)")
    t0 = time.time()

    diar_starts = [d.start for d in diar.segments]

    # Step 1: Transcribe whole audio as NL (fast)
    segs_gen, info_nl = model.transcribe(samples, language="nl", word_timestamps=True,
                                          beam_size=1, condition_on_previous_text=False,
                                          no_repeat_ngram_size=3, vad_filter=True)
    nl_segments = []
    for seg in segs_gen:
        text = seg.text.strip()
        if text:
            spk = find_speaker(seg.start, seg.end, diar.segments, diar_starts, names)
            nl_segments.append({
                "start": seg.start, "end": seg.end,
                "speaker": spk, "language": "nl", "text": text,
            })

    # Step 2: Detect language per 10s window
    window_langs = {}
    for wi in range(int(np.ceil(duration / 10))):
        ws = wi * 10
        we = min(ws + 10, duration)
        chunk = samples[int(ws * SAMPLE_RATE):int(we * SAMPLE_RATE)]
        if len(chunk) < SAMPLE_RATE * 0.3:
            continue
        # Quick transcribe just for language detection
        _, lang_info = model.transcribe(chunk, language=None, beam_size=1,
                                         word_timestamps=False, vad_filter=False,
                                         condition_on_previous_text=False)
        window_langs[wi] = lang_info.language

    non_nl_windows = {wi: lang for wi, lang in window_langs.items() if lang != "nl"}
    print(f"     Non-NL windows: {non_nl_windows}")

    # Step 3: Retranscribe non-NL windows with correct language
    retranscribed = {}
    for wi, lang in non_nl_windows.items():
        ws = wi * 10
        we = min(ws + 10, duration)
        chunk = samples[int(ws * SAMPLE_RATE):int(we * SAMPLE_RATE)]
        segs_gen, _ = model.transcribe(chunk, language=lang, word_timestamps=True,
                                        beam_size=1, condition_on_previous_text=False,
                                        no_repeat_ngram_size=3, vad_filter=False)
        for seg in segs_gen:
            text = seg.text.strip()
            if text:
                abs_start = seg.start + ws
                abs_end = seg.end + ws
                spk = find_speaker(abs_start, abs_end, diar.segments, diar_starts, names)
                retranscribed[abs_start] = {
                    "start": abs_start, "end": abs_end,
                    "speaker": spk, "language": lang, "text": text,
                }

    # Step 4: Merge — replace NL segments that fall in non-NL windows
    final_segments = []
    for seg in nl_segments:
        mid = (seg["start"] + seg["end"]) / 2
        window_idx = int(mid / 10)
        if window_idx in non_nl_windows:
            continue  # Skip — will be replaced by retranscribed
        final_segments.append(seg)

    # Add retranscribed segments
    final_segments.extend(retranscribed.values())
    final_segments.sort(key=lambda s: s["start"])

    elapsed = time.time() - t0
    rtf = elapsed / duration
    acc = measure_accuracy(final_segments)

    print(f"     ⏱  {elapsed:.1f}s (RTF {rtf:.3f}x)")
    print(f"     📊 Accuracy: {acc['accuracy']}% ({acc['correct']}/{acc['total']})")
    print(f"     🔄 Retranscribed: {len(retranscribed)} segments in {len(non_nl_windows)} windows")
    if acc["errors"]:
        for e in acc["errors"][:5]:
            print(f"     ❌ {e}")

    return {
        "name": name, "window_size": "hybrid",
        "elapsed_s": round(elapsed, 1), "rtf": round(rtf, 3),
        "n_segments": len(final_segments), "n_windows": f"{len(non_nl_windows)} retranscribed",
        "accuracy": acc["accuracy"], "correct": acc["correct"], "total": acc["total"],
        "segments": final_segments,
    }


# ═══════════════════════════════════════════════════════════
print(f"{'═'*60}")
print(f"  Autoresearch Run 3 — Multilingual Experiments")
print(f"  Test: 49s audio (Samuel NL + Evi FR)")
print(f"{'═'*60}")

clear_model_cache()
configure(model_size="large-v3-turbo", compute_type="int8", beam_size=1,
          condition_on_previous_text=False, no_repeat_ngram_size=3,
          temperature=(0.0, 0.2, 0.4), word_timestamps=True)

speaker_db = load_speaker_db()
print(f"  SpeakerDB loaded")

# Load audio
samples = load_audio_numpy()
duration = len(samples) / SAMPLE_RATE
print(f"  Audio: {duration:.1f}s")

# Diarize once (shared)
print(f"  Diarizing...")
diar = diarize(AUDIO, hf_token=HF_TOKEN, num_speakers=2)
names = speaker_db.identify(diar, threshold=0.30)
print(f"  Speakers: {names}")

# Load model
device = detect_device()
whisper_device = "auto" if device == "mps" else device
model = _get_model("large-v3-turbo", "int8", whisper_device, 0)

# ═══════════════════════════════════════════════════════════
# Experiments
# ═══════════════════════════════════════════════════════════

results = []

# ML4 Baseline: 10s windows
results.append(run_experiment("ML4-baseline-10s", 10.0, samples, model, diar, names, duration))

# ML5: 30s windows
results.append(run_experiment("ML5-30s-windows", 30.0, samples, model, diar, names, duration))

# ML5b: 20s windows
results.append(run_experiment("ML5b-20s-windows", 20.0, samples, model, diar, names, duration))

# ML5c: 15s windows
results.append(run_experiment("ML5c-15s-windows", 15.0, samples, model, diar, names, duration))

# ML5d: 5s windows (fine-grained)
results.append(run_experiment("ML5d-5s-windows", 5.0, samples, model, diar, names, duration))

# ML7: Hybrid (NL whole + retranscribe FR/EN)
results.append(run_hybrid("ML7-hybrid", samples, model, diar, names, duration))

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'╔'+'═'*58+'╗'}")
print(f"║  AUTORESEARCH RUN 3 — RESULTATEN                       ║")
print(f"{'╠'+'═'*58+'╣'}")
print(f"║  {'Experiment':<22} {'RTF':>8} {'Acc%':>6} {'Segs':>5} {'Win':>5}  ║")
print(f"{'╠'+'═'*58+'╣'}")

best_score = 0
best_name = ""
for r in results:
    score = r["accuracy"] / max(r["rtf"], 0.001)  # accuracy per RTF unit
    marker = ""
    if score > best_score:
        best_score = score
        best_name = r["name"]
        marker = " ⭐"
    wins = str(r["n_windows"])
    print(f"║  {r['name']:<22} {r['rtf']:>7.3f}x {r['accuracy']:>5.1f}% {r['n_segments']:>5} {wins:>5}{marker:<3}║")

print(f"{'╠'+'═'*58+'╣'}")
print(f"║  Beste: {best_name:<49}║")
print(f"{'╚'+'═'*58+'╝'}")

# Save
out = Path(__file__).parent / "autoresearch_ml_result.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"\n📄 Resultaat: {out}")
