"""Autoresearch Run 5 — Enhanced Speaker ID + Multilingual + Chunks.

Experiments:
  SI4-ECAPA — Enhanced speaker ID (pyannote+ECAPA+prosody vs pyannote-only)
  SI5      — num_speakers hint for diarization
  M2       — language="nl" explicit on FR/NL recordings
  C1       — chunk_length_s=15
  C2       — chunk_length_s=60
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_FILE = Path(__file__).parent / "autoresearch_results_run5.jsonl"
ECHO_ROOT = Path.home() / "Documents/Projecten/Plaude"
TRANSCRIPTY_ROOT = Path(__file__).parent.parent

start_time = time.time()
print(f"\n{'╔' + '═' * 58 + '╗'}")
print(f"║  Autoresearch Run 5 — Enhanced Speaker ID + More         ║")
print(f"║  Start: {time.strftime('%Y-%m-%d %H:%M')}                              ║")
print(f"{'╚' + '═' * 58 + '╝'}\n")


def log_result(exp_id, label, metrics, status, notes=""):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": 5, "exp": exp_id, "label": label,
        "metrics": metrics, "status": status, "notes": notes,
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Build Enhanced Speaker DB (ECAPA + Prosody for all speakers)
# ═══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PHASE 1: Build Enhanced Speaker DB")
print("=" * 60)

from transcripty.speakers import SpeakerDB, _cosine_similarity
from transcripty.ecapa import extract_ecapa_embedding
from transcripty.prosody import extract_prosodic_features

# Load pyannote embeddings from Echo DB
conn = sqlite3.connect(str(ECHO_ROOT / "echo.db"))
c = conn.cursor()
c.execute("""
    SELECT name, voiceprint, sample_path
    FROM speakers
    WHERE voiceprint IS NOT NULL AND is_active = 1
""")

enhanced_db = SpeakerDB()
speaker_audio_paths: dict[str, Path] = {}

for name, vp_json, sample_path in c.fetchall():
    pyannote_emb = json.loads(vp_json)

    # Resolve audio path
    if sample_path and not sample_path.startswith("/"):
        audio_path = ECHO_ROOT / sample_path
    elif sample_path:
        audio_path = Path(sample_path)
    else:
        audio_path = None

    speaker_audio_paths[name] = audio_path

    # Extract ECAPA embedding
    ecapa_emb = None
    prosodic_vec = None
    if audio_path and audio_path.exists():
        try:
            print(f"  {name}: extracting ECAPA + prosody from {audio_path.name}...")
            t0 = time.time()
            ecapa_emb = extract_ecapa_embedding(str(audio_path))
            prosodic = extract_prosodic_features(str(audio_path))
            prosodic_vec = prosodic.as_vector()
            dt = time.time() - t0
            print(f"    Done in {dt:.1f}s (ecapa={len(ecapa_emb)}d, prosody={len(prosodic_vec)}d)")
        except Exception as e:
            print(f"    ERROR: {e}")

    enhanced_db.enroll_from_embedding(
        name=name,
        embedding=pyannote_emb,
        ecapa_embedding=ecapa_emb,
        prosodic_features=prosodic_vec,
    )

conn.close()

# Override Samuel with multi-sample enhanced profile (from earlier today)
enhanced_samuel_path = TRANSCRIPTY_ROOT / "benchmarks/speakers_enhanced_samuel.json"
if enhanced_samuel_path.exists():
    samuel_db = SpeakerDB.load(str(enhanced_samuel_path))
    if "Samuel" in samuel_db.profiles:
        enhanced_db.profiles["Samuel"] = samuel_db.profiles["Samuel"]
        print(f"  Samuel: overridden with multi-sample enhanced profile")

# Save enhanced DB
enhanced_db_path = TRANSCRIPTY_ROOT / "benchmarks/speakers_enhanced_full.json"
enhanced_db.save(str(enhanced_db_path))

n_ecapa = sum(1 for p in enhanced_db.profiles.values() if p.ecapa_embedding)
n_prosody = sum(1 for p in enhanced_db.profiles.values() if p.prosodic_features)
print(f"\n  Enhanced DB: {len(enhanced_db)} speakers, {n_ecapa} with ECAPA, {n_prosody} with prosody")
print(f"  Saved to: {enhanced_db_path}")

# Also keep pyannote-only DB for baseline comparison
pyannote_only_db = SpeakerDB.load(str(TRANSCRIPTY_ROOT / "benchmarks/speakers_plaude.json"))


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: SI4-ECAPA — Enhanced vs Pyannote-only identification
# ═══════════════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("PHASE 2: SI4-ECAPA — Enhanced Speaker Identification")
print("=" * 60)

from transcripty.diarize import diarize
from transcripty.ecapa import extract_ecapa_embeddings_for_segments
from transcripty.prosody import extract_prosodic_for_segments

TEST_FILES = [
    {
        "id": 53,
        "path": ECHO_ROOT / "storage/2/e20d4d395b1d784ce6b6e4994c23887b.mp3",
        "label": "Roos gesprek",
        "duration": 191,
        "expected_speakers": ["Samuel", "Roos"],
    },
    {
        "id": 47,
        "path": ECHO_ROOT / "storage/2/3e156ded9f471f1297962d37dce3c845.mp3",
        "label": "audio NL",
        "duration": 313,
        "expected_speakers": ["Samuel"],
    },
]

si4_results = []

for tf in TEST_FILES:
    if not tf["path"].exists():
        print(f"\n  SKIP {tf['label']} — not found")
        continue

    print(f"\n{'─' * 60}")
    print(f"Test: {tf['label']} ({tf['duration']}s)")
    print(f"Expected: {tf['expected_speakers']}")
    print(f"{'─' * 60}")

    # Step 1: Diarize
    t0 = time.time()
    diar_result = diarize(audio_path=str(tf["path"]))
    diar_time = time.time() - t0
    n_spk = len(diar_result.embeddings) if diar_result.embeddings else 0
    print(f"  Diarization: {diar_time:.1f}s, {n_spk} speakers detected")

    if not diar_result.embeddings:
        continue

    # Step 2: Extract ECAPA embeddings for detected speakers
    # Build speaker_segments dict from diarization
    speaker_segs: dict[str, list[tuple[float, float]]] = {}
    for seg in diar_result.segments:
        label = seg.speaker
        if label not in speaker_segs:
            speaker_segs[label] = []
        speaker_segs[label].append((seg.start, seg.end))

    print(f"  Extracting ECAPA embeddings for {n_spk} speakers...")
    t0 = time.time()
    ecapa_embs = extract_ecapa_embeddings_for_segments(str(tf["path"]), speaker_segs)
    ecapa_time = time.time() - t0
    print(f"  ECAPA extraction: {ecapa_time:.1f}s")

    print(f"  Extracting prosodic features for {n_spk} speakers...")
    t0 = time.time()
    prosodic_feats = extract_prosodic_for_segments(str(tf["path"]), speaker_segs)
    prosody_time = time.time() - t0
    # Convert ProsodicFeatures to vectors
    prosodic_vecs = {
        spk: feats.as_vector() for spk, feats in prosodic_feats.items()
    }
    print(f"  Prosody extraction: {prosody_time:.1f}s")

    # Step 3: Compare identification methods
    # A) Pyannote-only (baseline)
    matches_pyannote = pyannote_only_db.identify(diar_result, threshold=0.40)
    # B) Enhanced (pyannote + ECAPA + prosody)
    matches_enhanced = enhanced_db.identify(
        diar_result,
        threshold=0.35,
        ecapa_embeddings=ecapa_embs,
        prosodic_features=prosodic_vecs,
        weights=(0.4, 0.4, 0.2),
    )

    print(f"\n  Pyannote-only (t=0.40): {len(matches_pyannote)}/{n_spk}")
    for lbl, name in matches_pyannote.items():
        correct = "✅" if name in tf["expected_speakers"] else "❌"
        print(f"    {correct} {lbl} → {name}")
    for lbl in diar_result.embeddings:
        if lbl not in matches_pyannote:
            print(f"    ⚪ {lbl} → UNKNOWN")

    print(f"\n  Enhanced (t=0.35, 40/40/20): {len(matches_enhanced)}/{n_spk}")
    for lbl, name in matches_enhanced.items():
        correct = "✅" if name in tf["expected_speakers"] else "❌"
        print(f"    {correct} {lbl} → {name}")
    for lbl in diar_result.embeddings:
        if lbl not in matches_enhanced:
            print(f"    ⚪ {lbl} → UNKNOWN")

    # Score: count correct matches
    correct_pyannote = sum(1 for n in matches_pyannote.values() if n in tf["expected_speakers"])
    correct_enhanced = sum(1 for n in matches_enhanced.values() if n in tf["expected_speakers"])
    false_pyannote = sum(1 for n in matches_pyannote.values() if n not in tf["expected_speakers"])
    false_enhanced = sum(1 for n in matches_enhanced.values() if n not in tf["expected_speakers"])

    si4_results.append({
        "file": tf["label"],
        "n_speakers": n_spk,
        "pyannote_correct": correct_pyannote,
        "pyannote_false": false_pyannote,
        "pyannote_total": len(matches_pyannote),
        "enhanced_correct": correct_enhanced,
        "enhanced_false": false_enhanced,
        "enhanced_total": len(matches_enhanced),
        "ecapa_time": ecapa_time,
        "prosody_time": prosody_time,
    })

# SI4 verdict
if si4_results:
    total_pyannote_correct = sum(r["pyannote_correct"] for r in si4_results)
    total_enhanced_correct = sum(r["enhanced_correct"] for r in si4_results)
    total_pyannote_false = sum(r["pyannote_false"] for r in si4_results)
    total_enhanced_false = sum(r["enhanced_false"] for r in si4_results)

    si4_status = "KEEP" if (
        total_enhanced_correct > total_pyannote_correct
        or (total_enhanced_correct == total_pyannote_correct and total_enhanced_false < total_pyannote_false)
    ) else "DISCARD"

    print(f"\n  {'═' * 50}")
    print(f"  SI4-ECAPA VERDICT: {si4_status}")
    print(f"  Pyannote-only: {total_pyannote_correct} correct, {total_pyannote_false} false positives")
    print(f"  Enhanced:      {total_enhanced_correct} correct, {total_enhanced_false} false positives")
    if si4_status == "KEEP":
        print(f"  ✅ Enhanced identification is BETER")
    print(f"  {'═' * 50}")

    log_result("SI4", "ecapa_enhanced_identification", {
        "pyannote_correct": total_pyannote_correct,
        "enhanced_correct": total_enhanced_correct,
        "pyannote_false": total_pyannote_false,
        "enhanced_false": total_enhanced_false,
        "details": si4_results,
    }, si4_status)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: SI5 — num_speakers hint
# ═══════════════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("PHASE 3: SI5 — num_speakers hint")
print("=" * 60)

si5_file = TEST_FILES[1]  # audio NL, 2 speakers
if si5_file["path"].exists():
    print(f"\nTest: {si5_file['label']} with num_speakers=2")

    t0 = time.time()
    diar_hinted = diarize(audio_path=str(si5_file["path"]), num_speakers=2)
    dt = time.time() - t0
    n_spk_hinted = len(diar_hinted.embeddings) if diar_hinted.embeddings else 0
    n_segs_hinted = len(diar_hinted.segments)

    # Compare with unhinted
    t0 = time.time()
    diar_auto = diarize(audio_path=str(si5_file["path"]))
    dt2 = time.time() - t0
    n_spk_auto = len(diar_auto.embeddings) if diar_auto.embeddings else 0
    n_segs_auto = len(diar_auto.segments)

    print(f"  Auto:       {n_spk_auto} speakers, {n_segs_auto} segments ({dt2:.1f}s)")
    print(f"  Hinted (2): {n_spk_hinted} speakers, {n_segs_hinted} segments ({dt:.1f}s)")

    # Test identification on both
    matches_auto = enhanced_db.identify(diar_auto, threshold=0.40)
    matches_hinted = enhanced_db.identify(diar_hinted, threshold=0.40)

    print(f"\n  Auto identification:   {matches_auto}")
    print(f"  Hinted identification: {matches_hinted}")

    si5_status = "KEEP" if len(matches_hinted) >= len(matches_auto) and n_spk_hinted == 2 else "DISCARD"
    print(f"\n  SI5 VERDICT: {si5_status}")

    log_result("SI5", "num_speakers_hint", {
        "auto_speakers": n_spk_auto, "hinted_speakers": n_spk_hinted,
        "auto_matches": len(matches_auto), "hinted_matches": len(matches_hinted),
    }, si5_status)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: M2 — Multilingual test (FR/NL)
# ═══════════════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("PHASE 4: M2 — Multilingual FR/NL test")
print("=" * 60)

# The 49s FR/NL test fragment
ML_TEST = ECHO_ROOT / "storage/2/69f151e0d62f1e694e5e9d878a02badb.mp3"

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

if ML_TEST.exists():
    # A) language="nl", multilingual=False (current speed config)
    print("\n  A) language='nl', multilingual=False (speed config)...")
    t0 = time.time()
    r_nl = transcribe(str(ML_TEST), **BEST_CONFIG, language="nl", multilingual=False)
    dt_nl = time.time() - t0
    print(f"     {len(r_nl.segments)} segments, {dt_nl:.1f}s, lang={r_nl.language}")

    # B) multilingual=True, language=None (auto-detect per chunk)
    print("\n  B) multilingual=True, language=None (auto-detect)...")
    t0 = time.time()
    r_ml = transcribe(str(ML_TEST), **BEST_CONFIG, language=None, multilingual=True)
    dt_ml = time.time() - t0
    print(f"     {len(r_ml.segments)} segments, {dt_ml:.1f}s, lang={r_ml.language}")

    # C) language="nl", multilingual=True (force NL but detect per chunk)
    print("\n  C) language='nl', multilingual=True...")
    t0 = time.time()
    r_nlml = transcribe(str(ML_TEST), **BEST_CONFIG, language="nl", multilingual=True)
    dt_nlml = time.time() - t0
    print(f"     {len(r_nlml.segments)} segments, {dt_nlml:.1f}s, lang={r_nlml.language}")

    # Show first few segments of each for quality comparison
    print("\n  Sample segments (first 5 of each):")
    for label, result in [("NL-only", r_nl), ("Auto-detect", r_ml), ("NL+multilingual", r_nlml)]:
        print(f"\n  [{label}]:")
        for seg in result.segments[:5]:
            lang_tag = f" [{seg.language}]" if seg.language else ""
            print(f"    {seg.start:.1f}-{seg.end:.1f}{lang_tag}: {seg.text[:80]}")

    # RTF comparison
    audio_dur = 49  # seconds
    rtf_nl = round(dt_nl / audio_dur, 3)
    rtf_ml = round(dt_ml / audio_dur, 3)
    rtf_nlml = round(dt_nlml / audio_dur, 3)

    print(f"\n  RTF comparison:")
    print(f"    NL-only:        {rtf_nl:.3f}x")
    print(f"    Auto-detect:    {rtf_ml:.3f}x")
    print(f"    NL+multilingual: {rtf_nlml:.3f}x")

    log_result("M2", "multilingual_fr_nl", {
        "rtf_nl_only": rtf_nl, "rtf_auto_detect": rtf_ml, "rtf_nl_multilingual": rtf_nlml,
        "segments_nl": len(r_nl.segments), "segments_ml": len(r_ml.segments),
    }, "INFO")
else:
    print(f"  SKIP — FR/NL test file not found: {ML_TEST}")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: C1/C2 — Chunk Length experiments
# ═══════════════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("PHASE 5: C1/C2 — Chunk Length Experiments")
print("=" * 60)

from benchmarks.autoresearch_runner import run_suite

# Warm-up run
print("\n  Warm-up run (loading model)...")
_ = run_suite("WARMUP", **BEST_CONFIG, language="nl", multilingual=False)

# Baseline (default chunk_length = 30)
print("\n  Baseline (chunk_length=30, default)...")
b = run_suite("BASELINE_C", **BEST_CONFIG, language="nl", multilingual=False)
BASELINE_SCORE = b["combined_score"]
print(f"  BASELINE: RTF={b['avg_rtf']:.3f}x | halluc={b['avg_halluc_rate']:.1%} | score={BASELINE_SCORE:.4f}")

# C1: chunk_length=15
# Note: faster-whisper doesn't directly expose chunk_length_s via transcribe()
# It's set via the WhisperModel.transcribe() call. Let's check if our wrapper supports it.
# Looking at transcribe.py — it doesn't pass chunk_length through. We need to test via
# the model directly or skip this experiment.
# For now, we'll note this as "requires code change" and skip.
print(f"\n  C1/C2: chunk_length niet direct beschikbaar in transcribe() API")
print(f"  → Vereist aanpassing in transcribe.py om chunk_length door te geven")
print(f"  → Overgeslagen in deze run")

log_result("C1", "chunk_length_15", {"note": "requires transcribe.py API change"}, "SKIPPED")
log_result("C2", "chunk_length_60", {"note": "requires transcribe.py API change"}, "SKIPPED")


# ═══════════════════════════════════════════════════════════════════════
# EINDRAPPORT
# ═══════════════════════════════════════════════════════════════════════

elapsed_min = round((time.time() - start_time) / 60, 1)

print(f"\n\n{'╔' + '═' * 58 + '╗'}")
print(f"║  AUTORESEARCH RUN 5 EINDRAPPORT                        ║")
print(f"{'╠' + '═' * 58 + '╣'}")
print(f"║  Gestopt:   {time.strftime('%Y-%m-%d %H:%M')}                              ║")
print(f"║  Duur:      {elapsed_min} minuten                                  ║")
print(f"{'╚' + '═' * 58 + '╝'}\n")

print("Experimenten:")
print(f"{'─' * 60}")

if si4_results:
    print(f"  SI4-ECAPA: {'✅ KEEP' if si4_status == 'KEEP' else '❌ DISCARD'}")
    print(f"    Pyannote-only: {total_pyannote_correct} correct, {total_pyannote_false} false")
    print(f"    Enhanced:      {total_enhanced_correct} correct, {total_enhanced_false} false")

print(f"  SI5: num_speakers hint → {'✅ KEEP' if si5_status == 'KEEP' else '❌ DISCARD'}")
print(f"  M2:  multilingual FR/NL → INFO (kwaliteitsanalyse)")
print(f"  C1:  chunk_length=15 → SKIPPED (API change nodig)")
print(f"  C2:  chunk_length=60 → SKIPPED (API change nodig)")

print(f"\n📄 Log: benchmarks/autoresearch_results_run5.jsonl")
print(f"📦 Enhanced DB: benchmarks/speakers_enhanced_full.json")
print(f"⏱️  Totale duur: {elapsed_min} minuten")
