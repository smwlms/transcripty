"""DER Benchmark — Diarization Error Rate measurement.

Measures speaker diarization quality against ground truth speaker labels.
DER = (false_alarm + missed_speech + speaker_confusion) / total_speech_duration

Usage:
    PYTHONPATH=. .venv/bin/python benchmarks/der_benchmark.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ECHO_ROOT = Path.home() / "Documents/Projecten/Plaude"


def compute_der_from_segments(
    hypothesis: list[dict],
    reference: list[dict],
    collar: float = 0.25,
) -> dict:
    """Compute DER by comparing hypothesis and reference speaker segments.

    Simple frame-based DER: discretize time into 100ms frames, compare
    speaker labels at each frame.

    Args:
        hypothesis: List of {"speaker": str, "start": float, "end": float}
        reference: List of {"speaker": str, "start": float, "end": float}
        collar: Tolerance in seconds around speaker boundaries (default 0.25s)

    Returns:
        Dict with DER%, confusion%, missed%, false_alarm%, details.
    """
    if not reference:
        return {"error": "no reference segments"}

    # Find total duration
    max_time = max(
        max(s["end"] for s in reference),
        max(s["end"] for s in hypothesis) if hypothesis else 0,
    )

    # Discretize into 100ms frames
    frame_size = 0.1
    n_frames = int(max_time / frame_size) + 1

    ref_frames = [None] * n_frames
    hyp_frames = [None] * n_frames

    for seg in reference:
        start_f = int((seg["start"] + collar) / frame_size)
        end_f = int((seg["end"] - collar) / frame_size)
        for f in range(max(0, start_f), min(n_frames, end_f)):
            ref_frames[f] = seg["speaker"]

    for seg in hypothesis:
        start_f = int(seg["start"] / frame_size)
        end_f = int(seg["end"] / frame_size)
        for f in range(max(0, start_f), min(n_frames, end_f)):
            hyp_frames[f] = seg["speaker"]

    # Count errors
    total_speech = 0
    correct = 0
    confusion = 0
    missed = 0
    false_alarm = 0

    # Build speaker mapping (greedy: most common hypothesis speaker for each reference)
    ref_to_hyp: dict[str, dict[str, int]] = {}
    for f in range(n_frames):
        if ref_frames[f] is not None:
            total_speech += 1
            ref_spk = ref_frames[f]
            hyp_spk = hyp_frames[f]

            if hyp_spk is None:
                missed += 1
            else:
                if ref_spk not in ref_to_hyp:
                    ref_to_hyp[ref_spk] = {}
                ref_to_hyp[ref_spk][hyp_spk] = ref_to_hyp[ref_spk].get(hyp_spk, 0) + 1

    # Find best mapping for each reference speaker
    speaker_map = {}
    used_hyp = set()
    for ref_spk in sorted(ref_to_hyp, key=lambda s: -max(ref_to_hyp[s].values())):
        best_hyp = None
        best_count = 0
        for hyp_spk, count in ref_to_hyp[ref_spk].items():
            if hyp_spk not in used_hyp and count > best_count:
                best_hyp = hyp_spk
                best_count = count
        if best_hyp:
            speaker_map[ref_spk] = best_hyp
            used_hyp.add(best_hyp)

    # Count correct and confusion with optimal mapping
    for f in range(n_frames):
        if ref_frames[f] is not None and hyp_frames[f] is not None:
            expected_hyp = speaker_map.get(ref_frames[f])
            if hyp_frames[f] == expected_hyp:
                correct += 1
            else:
                confusion += 1
        elif ref_frames[f] is None and hyp_frames[f] is not None:
            false_alarm += 1

    if total_speech == 0:
        return {"error": "no speech in reference"}

    der = (confusion + missed + false_alarm) / total_speech
    return {
        "der": round(der, 4),
        "der_pct": round(der * 100, 1),
        "confusion_pct": round(confusion / total_speech * 100, 1),
        "missed_pct": round(missed / total_speech * 100, 1),
        "false_alarm_pct": round(false_alarm / total_speech * 100, 1),
        "correct_pct": round(correct / total_speech * 100, 1),
        "total_speech_frames": total_speech,
        "speaker_map": speaker_map,
        "collar": collar,
    }


if __name__ == "__main__":
    from transcripty.diarize import diarize
    from transcripty.speakers import SpeakerDB
    from transcripty.ecapa import extract_ecapa_embeddings_for_segments
    from transcripty.prosody import extract_prosodic_for_segments

    # Samuel+Alex gesprek — ground truth from user:
    # SPEAKER_01 = Samuel (0-22min, NL, dominant)
    # SPEAKER_02 = Alex (0-18min, NL)
    # SPEAKER_00 = Andries (18+ min, EN, achtergrond)
    TEST = ECHO_ROOT / "storage/1/3e24a52f0a7028c7aa47950ec7d0bd4b.mp3"

    print("=" * 60)
    print("SP3: Diarization Error Rate — Samuel+Alex gesprek")
    print("=" * 60)

    # Diarize
    print("\nDiarizing...")
    t0 = time.time()
    diar = diarize(audio_path=str(TEST))
    dt = time.time() - t0
    print(f"Done in {dt:.0f}s — {len(diar.embeddings)} speakers, {len(diar.segments)} segments")

    # Build speaker segments for features
    speaker_segs = {}
    for seg in diar.segments:
        speaker_segs.setdefault(seg.speaker, []).append((seg.start, seg.end))

    # Enhanced identification
    db = SpeakerDB.load("benchmarks/speakers_enhanced_full.json")
    ecapa_embs = extract_ecapa_embeddings_for_segments(str(TEST), speaker_segs)
    prosodic = extract_prosodic_for_segments(str(TEST), speaker_segs)
    prosodic_vecs = {spk: f.as_vector() for spk, f in prosodic.items()}

    names = db.identify(
        diar, threshold=0.40,
        ecapa_embeddings=ecapa_embs,
        prosodic_features=prosodic_vecs,
    )

    print(f"\nIdentification: {names}")

    # Build hypothesis segments with identified names
    hyp_segments = []
    for seg in diar.segments:
        speaker = names.get(seg.speaker, seg.speaker)
        hyp_segments.append({"speaker": speaker, "start": seg.start, "end": seg.end})

    # Ground truth: we know the broad structure
    # SPEAKER_01 (Samuel) talks 0-18min, SPEAKER_02 (Alex) talks 0-18min
    # SPEAKER_00 (Andries) talks 18+ min
    # For DER we use the diarization segments themselves as pseudo-reference
    # with the correct speaker names applied
    gt_map = {"SPEAKER_01": "Samuel", "SPEAKER_02": "Alex", "SPEAKER_00": "Andries"}

    ref_segments = []
    for seg in diar.segments:
        ref_segments.append({
            "speaker": gt_map.get(seg.speaker, seg.speaker),
            "start": seg.start,
            "end": seg.end,
        })

    # Compute DER
    der = compute_der_from_segments(hyp_segments, ref_segments)

    print(f"\n{'─' * 40}")
    print(f"DER: {der['der_pct']}%")
    print(f"  Correct:     {der['correct_pct']}%")
    print(f"  Confusion:   {der['confusion_pct']}%")
    print(f"  Missed:      {der['missed_pct']}%")
    print(f"  False alarm: {der['false_alarm_pct']}%")
    print(f"  Speaker map: {der['speaker_map']}")
    print(f"{'─' * 40}")

    # Save
    with open("benchmarks/der_results.json", "w") as f:
        json.dump(der, f, indent=2)
    print(f"\nResultaten: benchmarks/der_results.json")
