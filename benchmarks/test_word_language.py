import os
"""Test per-woord taaldetectie — diarize-first met gegroepeerde chunks.

Diarize EERST → groepeer consecutive speaker turns tot ~30s chunks →
Whisper per chunk met language=None → correcte taal per chunk.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from transcripty.multilingual import transcribe_with_speakers_multilingual
from transcripty.transcribe import clear_model_cache
from transcripty.config import configure
from transcripty.speakers import SpeakerDB

AUDIO = Path.home() / "Documents/Projecten/Plaude/storage/1/881a40bd2e0586268640f9d94efd56f9.mp3"
DURATION = 3706.8
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def load_speaker_db() -> SpeakerDB | None:
    try:
        sys.path.insert(0, str(Path.home() / "Documents/Projecten/Plaude"))
        from sqlalchemy import create_engine, text as sqlt
        engine = create_engine("sqlite:///" + str(Path.home() / "Documents/Projecten/Plaude/echo.db"))
        with engine.connect() as conn:
            speakers = conn.execute(sqlt(
                "SELECT name, voiceprint FROM speakers WHERE is_active=1 AND voiceprint IS NOT NULL"
            )).fetchall()
        if not speakers:
            return None
        db = SpeakerDB()
        for name, vp in speakers:
            embedding = json.loads(vp.decode("utf-8"))
            db.enroll_from_embedding(name, embedding)
        print(f"  SpeakerDB: {len(speakers)} sprekers geladen")
        return db
    except Exception as e:
        print(f"  SpeakerDB laden mislukt: {e}")
        return None


def progress_cb(pct: float, msg: str):
    print(f"  [{pct*100:5.1f}%] {msg}")


print(f"\n{'═'*62}")
print(f"  DIARIZE-FIRST per-woord taaldetectie (gegroepeerd)")
print(f"  881a40bd (61.8 min, NL/FR/EN, ~4 sprekers)")
print(f"{'═'*62}")

clear_model_cache()
configure(
    model_size="large-v3-turbo",
    compute_type="int8",
    condition_on_previous_text=False,
    no_repeat_ngram_size=3,
    beam_size=1,
    word_timestamps=True,
    temperature=(0.0, 0.2, 0.4),
)

speaker_db = load_speaker_db()
t0 = time.time()

segments = transcribe_with_speakers_multilingual(
    AUDIO,
    hf_token=HF_TOKEN,
    speaker_db=speaker_db,
    speaker_threshold=0.30,
    num_speakers=4,
    on_progress=progress_cb,
    # Whisper kwargs
    beam_size=1,
    word_timestamps=True,
    condition_on_previous_text=False,
    no_repeat_ngram_size=3,
    temperature=(0.0, 0.2, 0.4),
)

elapsed = time.time() - t0
rtf = elapsed / DURATION

# Statistieken
lang_word_counts: Counter = Counter()
lang_seg_counts: Counter = Counter()
speaker_counts: Counter = Counter()
fr_segments = []
en_segments = []

for seg in segments:
    lang_seg_counts[seg.language or "?"] += 1
    speaker_counts[seg.speaker] += 1
    for w in seg.words:
        lang_word_counts[w.language or "?"] += 1
    if seg.language == "fr":
        fr_segments.append(seg)
    elif seg.language == "en":
        en_segments.append(seg)

total_words = sum(lang_word_counts.values())
total_segs = sum(lang_seg_counts.values())

print(f"\n  ⏱  Tijd:      {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"  🚀 RTF:       {rtf:.3f}x realtime")
print(f"  📝 Segmenten: {len(segments)}")

print(f"\n  📊 Taalverdeling per SEGMENT:")
for lang, count in lang_seg_counts.most_common():
    pct = round(count * 100 / total_segs) if total_segs else 0
    print(f"     {lang}: {count} ({pct}%)")

print(f"\n  📊 Taalverdeling per WOORD:")
for lang, count in lang_word_counts.most_common():
    pct = round(count * 100 / total_words) if total_words else 0
    print(f"     {lang}: {count} ({pct}%)")

print(f"\n  👥 Sprekers: {dict(speaker_counts.most_common())}")

if fr_segments:
    print(f"\n  🇫🇷 Franse segmenten ({len(fr_segments)}):")
    for seg in fr_segments[:30]:
        print(f"     {seg.start:7.1f}s  [{seg.speaker}]  {seg.text.strip()[:100]}")

if en_segments:
    print(f"\n  🇬🇧 Engelse segmenten ({len(en_segments)}):")
    for seg in en_segments[:15]:
        print(f"     {seg.start:7.1f}s  [{seg.speaker}]  {seg.text.strip()[:100]}")

print(f"\n  --- Eerste 40 segmenten ---")
for seg in segments[:40]:
    lang_tag = f"[{seg.language}]" if seg.language else "[?]"
    print(f"  {seg.start:7.1f}s  [{seg.speaker}]  {lang_tag}  {seg.text.strip()[:80]}")

# Sla op
out = Path(__file__).parent / "word_language_result.json"
data = []
for seg in segments:
    data.append({
        "start": seg.start,
        "end": seg.end,
        "speaker": seg.speaker,
        "language": seg.language,
        "text": seg.text,
        "words": [
            {"text": w.text, "start": w.start, "end": w.end, "language": w.language}
            for w in seg.words
        ],
    })

with open(out, "w") as f:
    json.dump({
        "rtf": round(rtf, 3),
        "elapsed_s": round(elapsed, 1),
        "lang_segments": dict(lang_seg_counts),
        "lang_words": dict(lang_word_counts),
        "speakers": dict(speaker_counts),
        "segments": data,
    }, f, indent=2, ensure_ascii=False)

print(f"\n📄 Resultaat: {out}")
