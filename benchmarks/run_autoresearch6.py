"""Autoresearch Run 6 — WER3, WER4, QF1, QF2, ML9.

Code-based experiments that don't require ground truth input.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ECHO_ROOT = Path.home() / "Documents/Projecten/Plaude"
RESULTS_FILE = Path(__file__).parent / "autoresearch_results_run6.jsonl"

start_time = time.time()
print(f"\n{'╔' + '═' * 58 + '╗'}")
print(f"║  Autoresearch Run 6 — WER + Confidence + Multilingual    ║")
print(f"║  Start: {time.strftime('%Y-%m-%d %H:%M')}                              ║")
print(f"{'╚' + '═' * 58 + '╝'}\n")


def log_result(exp_id, label, metrics, status, notes=""):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": 6, "exp": exp_id, "label": label,
        "metrics": metrics, "status": status, "notes": notes,
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


GT = "U spreekt met Samuel van Claes en Willems. Ik bel u even op naar aanleiding van uw interesse in het pand aan de Hoogstraat. Het betreft een rijwoning met drie slaapkamers, een EPC-label B en een kadastraal inkomen van 850 euro. De vraagprijs is 325.000 euro. Er is al een bod binnengekomen, maar de opschortende voorwaarden zijn nog niet vervuld. De notaris heeft bevestigd dat de stedenbouwkundige vergunning in orde is. Wenst u een bezoek in te plannen deze week?"
SAMPLE4 = ECHO_ROOT / "storage/1/40527e4d98a6bd5ef5455401c5d28c96.mp3"


def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = d[i - 1][j - 1] if r[i - 1] == h[j - 1] else 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return round(d[n][m] / n * 100, 1)


# ═══════════════════════════════════════════════════════════════════════
# WER3: Post-processing correcties
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("WER3: Post-processing correcties")
print("=" * 60)

from transcripty import transcribe
from transcripty.postprocess import correct_text

P2 = dict(
    model_size="large-v3-turbo", compute_type="int8", beam_size=3,
    word_timestamps=True, temperature=0.0, vad_filter=True,
    condition_on_previous_text=False, hallucination_silence_threshold=2.0,
    no_repeat_ngram_size=3, language=None, multilingual=True,
)

if SAMPLE4.exists():
    r = transcribe(str(SAMPLE4), **P2)
    raw_text = " ".join(seg.text for seg in r.segments)
    fixed_text = correct_text(raw_text)

    wer_raw = wer(GT, raw_text)
    wer_fixed = wer(GT, fixed_text)

    print(f"  Raw WER:   {wer_raw}%")
    print(f"  Fixed WER: {wer_fixed}%")
    print(f"  Delta:     {wer_fixed - wer_raw:+.1f}%")

    if wer_fixed < wer_raw:
        print(f"\n  Correcties toegepast:")
        raw_words = raw_text.split()
        fixed_words = fixed_text.split()
        for rw, fw in zip(raw_words, fixed_words):
            if rw != fw:
                print(f"    {rw} → {fw}")

    status = "KEEP" if wer_fixed < wer_raw else "DISCARD"
    log_result("WER3", "postprocess", {"wer_raw": wer_raw, "wer_fixed": wer_fixed}, status)
    print(f"\n  WER3: {status}")


# ═══════════════════════════════════════════════════════════════════════
# WER4: P2 turbo vs large-v3 WER vergelijking
# ═══════════════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("WER4: P2 turbo vs large-v3")
print("=" * 60)

if SAMPLE4.exists():
    configs = [
        ("P2_turbo", dict(P2)),
        ("P2_largev3", dict(P2, model_size="large-v3")),
    ]

    for label, cfg in configs:
        t0 = time.time()
        r = transcribe(str(SAMPLE4), **cfg)
        dt = time.time() - t0
        text = " ".join(seg.text for seg in r.segments)
        fixed = correct_text(text)
        w_raw = wer(GT, text)
        w_fix = wer(GT, fixed)
        rtf = round(dt / 34, 3)

        print(f"\n  {label}: WER={w_raw}% (raw), {w_fix}% (fixed), RTF={rtf}x")
        print(f"    Text: {fixed[:120]}")
        log_result("WER4", label, {"wer_raw": w_raw, "wer_fixed": w_fix, "rtf": rtf}, "INFO")


# ═══════════════════════════════════════════════════════════════════════
# QF1: Confidence scores extraheren
# ═══════════════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("QF1: Confidence scores per segment")
print("=" * 60)

# Use a longer file for more diverse segments
LONG_FILE = ECHO_ROOT / "storage/2/e20d4d395b1d784ce6b6e4994c23887b.mp3"  # 191s

if LONG_FILE.exists():
    r = transcribe(str(LONG_FILE), **P2)

    # Analyze confidence distribution
    logprobs = [s.avg_logprob for s in r.segments if s.avg_logprob is not None]
    no_speech = [s.no_speech_prob for s in r.segments if s.no_speech_prob is not None]

    if logprobs:
        import statistics
        print(f"  Segmenten: {len(r.segments)}")
        print(f"  avg_logprob: min={min(logprobs):.3f}, max={max(logprobs):.3f}, "
              f"mean={statistics.mean(logprobs):.3f}, stdev={statistics.stdev(logprobs):.3f}")
        print(f"  no_speech_prob: min={min(no_speech):.3f}, max={max(no_speech):.3f}, "
              f"mean={statistics.mean(no_speech):.3f}")

        # Flag low-confidence segments
        LOW_CONF_THRESHOLD = -1.0  # avg_logprob below this = suspicious
        HIGH_NOSPEECH = 0.5  # no_speech_prob above this = possibly silence

        suspicious = []
        for seg in r.segments:
            if seg.avg_logprob is not None and seg.avg_logprob < LOW_CONF_THRESHOLD:
                suspicious.append(("low_logprob", seg))
            elif seg.no_speech_prob is not None and seg.no_speech_prob > HIGH_NOSPEECH:
                suspicious.append(("high_nospeech", seg))

        print(f"\n  Verdachte segmenten (logprob<{LOW_CONF_THRESHOLD} of nospeech>{HIGH_NOSPEECH}): {len(suspicious)}/{len(r.segments)}")
        for reason, seg in suspicious[:5]:
            print(f"    [{reason}] {seg.start:.1f}-{seg.end:.1f} logprob={seg.avg_logprob:.3f} "
                  f"nospeech={seg.no_speech_prob:.3f}: {seg.text[:60]}")

        log_result("QF1", "confidence_analysis", {
            "n_segments": len(r.segments),
            "logprob_mean": round(statistics.mean(logprobs), 3),
            "logprob_stdev": round(statistics.stdev(logprobs), 3),
            "nospeech_mean": round(statistics.mean(no_speech), 3),
            "n_suspicious": len(suspicious),
            "threshold_logprob": LOW_CONF_THRESHOLD,
            "threshold_nospeech": HIGH_NOSPEECH,
        }, "KEEP")
    else:
        print("  Geen confidence scores beschikbaar (older faster-whisper?)")
        log_result("QF1", "confidence_analysis", {"error": "no logprobs"}, "SKIPPED")


# ═══════════════════════════════════════════════════════════════════════
# ML9: Whisper audio-detect vs lingua tekst-detect
# ═══════════════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("ML9: Whisper audio-detect vs lingua tekst-detect")
print("=" * 60)

FR_FILE = ECHO_ROOT / "storage/2/241618a9fd7e2b43868358b12cb43189.mp3"  # Visite immobilière 1316s

if FR_FILE.exists():
    # Method A: multilingual=True (Whisper per-chunk detect) + lingua post-processing
    print("\n  A) Whisper multilingual + lingua post-process...")
    t0 = time.time()
    r_a = transcribe(str(FR_FILE), **P2)
    dt_a = time.time() - t0
    langs_a = {}
    for seg in r_a.segments:
        l = seg.language or "?"
        langs_a[l] = langs_a.get(l, 0) + 1
    print(f"     {len(r_a.segments)} segs, RTF={dt_a/1316:.3f}x, langs={langs_a}")

    # Method B: Whisper with language=None (global detect only, no lingua)
    # We need to skip lingua post-processing
    print("\n  B) Whisper global detect only (no lingua)...")
    cfg_b = dict(P2, multilingual=False, language=None)
    t0 = time.time()
    r_b = transcribe(str(FR_FILE), **cfg_b)
    dt_b = time.time() - t0
    langs_b = {}
    for seg in r_b.segments:
        l = seg.language or "?"
        langs_b[l] = langs_b.get(l, 0) + 1
    print(f"     {len(r_b.segments)} segs, RTF={dt_b/1316:.3f}x, langs={langs_b}")

    # Compare first 10 segments
    print(f"\n  Vergelijking eerste 10 segmenten:")
    print(f"  {'#':>3} {'A (lingua)':>10} {'B (global)':>10} {'Text A':>50}")
    for i, (sa, sb) in enumerate(zip(r_a.segments[:10], r_b.segments[:10])):
        la = sa.language or "?"
        lb = sb.language or "?"
        match = "✅" if la == lb else "❌"
        print(f"  {i:>3} {la:>10} {lb:>10} {match} {sa.text[:45]}")

    log_result("ML9", "whisper_vs_lingua", {
        "method_a_langs": langs_a,
        "method_b_langs": langs_b,
        "method_a_rtf": round(dt_a / 1316, 3),
        "method_b_rtf": round(dt_b / 1316, 3),
    }, "INFO")
else:
    print("  SKIP — FR/NL file niet gevonden")


# ═══════════════════════════════════════════════════════════════════════
# EINDRAPPORT
# ═══════════════════════════════════════════════════════════════════════
elapsed_min = round((time.time() - start_time) / 60, 1)

print(f"\n\n{'╔' + '═' * 58 + '╗'}")
print(f"║  AUTORESEARCH RUN 6 EINDRAPPORT                        ║")
print(f"{'╠' + '═' * 58 + '╣'}")
print(f"║  Duur: {elapsed_min} minuten                                    ║")
print(f"{'╚' + '═' * 58 + '╝'}")
print(f"\n📄 Log: {RESULTS_FILE}")
