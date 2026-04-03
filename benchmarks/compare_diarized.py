import os
"""Turbo vs Large-v3 — volledige vergelijking MET diarization.

Bestand: 881a40bd (61.8 min, NL/FR/EN mix, ~4 sprekers)
Toont: RTF, taalverdeling, sprekers, kwaliteit eerste segmenten.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transcripty import transcribe_with_speakers
from transcripty.diarize import diarize
from transcripty.formatters import to_text
from transcripty.transcribe import clear_model_cache
from transcripty.config import configure
from transcripty.speakers import SpeakerDB

AUDIO = Path.home() / "Documents/Projecten/Plaude/storage/1/881a40bd2e0586268640f9d94efd56f9.mp3"
DURATION = 3706.8
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Speakers uit Echo DB
SPEAKERS_FILE = Path(__file__).parent / "speakers.json"

COMMON_TRANSCRIBE = dict(
    vad_filter=True,
    condition_on_previous_text=False,
    no_repeat_ngram_size=3,
    beam_size=1,
    word_timestamps=True,
    multilingual=True,
    language=None,
    temperature=(0.0, 0.2, 0.4),
)


def load_speaker_db() -> SpeakerDB | None:
    """Laad speaker DB vanuit Echo database."""
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path.home() / "Documents/Projecten/Plaude"))
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
        print(f"  SpeakerDB: {len(speakers)} sprekers geladen ({', '.join(s[0] for s in speakers)})")
        return db
    except Exception as e:
        print(f"  SpeakerDB laden mislukt: {e}")
        return None


def run(model_label: str, model_size: str):
    print(f"\n{'═'*62}")
    print(f"  Model: {model_label}  |  diarization: ON  |  multilingual: ON")
    print(f"{'═'*62}")

    clear_model_cache()
    configure(
        model_size=model_size,
        compute_type="int8",
        **{k: v for k, v in COMMON_TRANSCRIBE.items()},
    )

    speaker_db = load_speaker_db()
    t0 = time.time()

    segments = transcribe_with_speakers(
        AUDIO,
        hf_token=HF_TOKEN,
        speaker_db=speaker_db,
        speaker_threshold=0.30,
        num_speakers=4,
    )

    elapsed = time.time() - t0
    rtf = elapsed / DURATION

    # Taalverdeling
    lang_counts: dict[str, int] = {}
    for s in segments:
        lg = getattr(s, "language", None)
        if lg:
            lang_counts[lg] = lang_counts.get(lg, 0) + 1
    total_lg = sum(lang_counts.values())
    lang_pct = {k: round(v*100/total_lg) for k, v in sorted(lang_counts.items(), key=lambda x: -x[1])} if total_lg else {}

    # Sprekers
    speaker_counts: dict[str, int] = {}
    for s in segments:
        sp = s.speaker or "UNKNOWN"
        speaker_counts[sp] = speaker_counts.get(sp, 0) + 1

    print(f"\n  ⏱  Tijd:      {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  🚀 RTF:       {rtf:.3f}x realtime")
    print(f"  📝 Segmenten: {len(segments)}")
    print(f"  🌍 Talen:     {lang_pct if lang_pct else 'geen per-segment detectie'}")
    print(f"  👥 Sprekers:  {dict(sorted(speaker_counts.items(), key=lambda x: -x[1]))}")

    print(f"\n  --- Eerste 25 segmenten ---")
    for s in segments[:25]:
        lang_tag = f"[{s.language}] " if getattr(s, "language", None) else ""
        spk = s.speaker or "?"
        print(f"  {s.start:6.1f}s  [{spk}]  {lang_tag}{s.text.strip()}")

    return {
        "model": model_label,
        "elapsed_s": round(elapsed, 1),
        "rtf": round(rtf, 3),
        "segments": len(segments),
        "languages": lang_pct,
        "speakers": speaker_counts,
        "text_sample": to_text(segments[:80]),
    }


print(f"\n{'╔'+'═'*60+'╗'}")
print(f"║  Turbo vs Large-v3 — WITH diarization                    ║")
print(f"║  881a40bd | 61.8 min | NL/FR/EN | ~4 sprekers             ║")
print(f"{'╚'+'═'*60+'╝'}")

result_turbo = run("large-v3-turbo", "large-v3-turbo")
result_large = run("large-v3", "large-v3")

# Vergelijking
speedup = round(result_large["rtf"] / result_turbo["rtf"], 1)
print(f"\n\n{'╔'+'═'*60+'╗'}")
print(f"║  VERGELIJKING                                              ║")
print(f"{'╠'+'═'*60+'╣'}")
print(f"║  Turbo:  {result_turbo['rtf']:.3f}x RTF  ({result_turbo['elapsed_s']/60:.1f} min totaal)             ║")
print(f"║  Large:  {result_large['rtf']:.3f}x RTF  ({result_large['elapsed_s']/60:.1f} min totaal)            ║")
print(f"║  Turbo is {speedup}x sneller                                  ║")
print(f"║                                                            ║")
print(f"║  Turbo talen: {result_turbo['languages']}                        ║")
print(f"║  Large talen: {result_large['languages']}                        ║")
print(f"{'╚'+'═'*60+'╝'}")

out = Path(__file__).parent / "compare_diarized_result.json"
with open(out, "w") as f:
    json.dump({"turbo": result_turbo, "large": result_large}, f, indent=2, ensure_ascii=False)
print(f"\n📄 Log: {out}")
