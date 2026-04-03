"""Quality fallback — retranscribe low-confidence segments.

Detects segments with low avg_logprob (Whisper's internal confidence)
and retranscribes them with more conservative settings (higher beam size).

Usage:
    from transcripty.quality_fallback import retranscribe_low_confidence
    improved = retranscribe_low_confidence(result, audio_path)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from transcripty.models import Segment, TranscriptionResult

logger = logging.getLogger(__name__)

# Thresholds determined from QF1 analysis (Run 6):
# Good segments: avg_logprob between -0.5 and -0.2
# Suspicious: avg_logprob < -0.8
DEFAULT_LOGPROB_THRESHOLD = -0.8
DEFAULT_NOSPEECH_THRESHOLD = 0.6


def find_low_confidence_segments(
    result: TranscriptionResult,
    logprob_threshold: float = DEFAULT_LOGPROB_THRESHOLD,
    nospeech_threshold: float = DEFAULT_NOSPEECH_THRESHOLD,
) -> list[int]:
    """Find indices of segments with low confidence scores.

    Args:
        result: TranscriptionResult with avg_logprob populated.
        logprob_threshold: Segments below this are suspicious.
        nospeech_threshold: Segments above this no_speech_prob are suspicious.

    Returns:
        List of segment indices that should be retranscribed.
    """
    suspicious = []
    for i, seg in enumerate(result.segments):
        if seg.avg_logprob is not None and seg.avg_logprob < logprob_threshold:
            suspicious.append(i)
        elif seg.no_speech_prob is not None and seg.no_speech_prob > nospeech_threshold:
            suspicious.append(i)
    return suspicious


def retranscribe_low_confidence(
    result: TranscriptionResult,
    audio_path: str | Path,
    logprob_threshold: float = DEFAULT_LOGPROB_THRESHOLD,
    nospeech_threshold: float = DEFAULT_NOSPEECH_THRESHOLD,
    fallback_beam_size: int = 5,
) -> TranscriptionResult:
    """Retranscribe segments with low confidence using higher beam size.

    Only retranscribes the time ranges of suspicious segments, not the
    entire audio. This keeps the overhead minimal.

    Args:
        result: Original TranscriptionResult.
        audio_path: Path to the audio file.
        logprob_threshold: Segments below this avg_logprob are retranscribed.
        nospeech_threshold: Segments above this no_speech_prob are retranscribed.
        fallback_beam_size: Beam size for retranscription (default 5).

    Returns:
        TranscriptionResult with suspicious segments replaced by retranscribed versions.
        Segments that improved (higher avg_logprob) are kept; others retain the original.
    """
    suspicious_indices = find_low_confidence_segments(
        result, logprob_threshold, nospeech_threshold,
    )

    if not suspicious_indices:
        logger.info("No low-confidence segments found (threshold=%.2f)", logprob_threshold)
        return result

    logger.info(
        "Found %d low-confidence segments (logprob<%.2f or nospeech>%.2f). Retranscribing...",
        len(suspicious_indices), logprob_threshold, nospeech_threshold,
    )

    from transcripty.audio import wav_audio
    from transcripty.config import get_config
    from transcripty.transcribe import _get_model
    from transcripty.device import detect_device

    cfg = get_config()
    device = detect_device()
    whisper_device = "auto" if device == "mps" else device
    model = _get_model(cfg.model_size, cfg.compute_type, whisper_device, cfg.cpu_threads)

    audio_path = Path(audio_path)
    improved_count = 0

    with wav_audio(audio_path) as wav_path:
        for idx in suspicious_indices:
            seg = result.segments[idx]
            original_logprob = seg.avg_logprob or -999

            # Retranscribe just this time range with higher beam
            try:
                segments_gen, info = model.transcribe(
                    str(wav_path),
                    beam_size=fallback_beam_size,
                    language=cfg.language,
                    multilingual=cfg.multilingual,
                    word_timestamps=cfg.word_timestamps,
                    vad_filter=cfg.vad_filter,
                    condition_on_previous_text=False,
                    no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                    temperature=0.0,
                    clip_timestamps=[seg.start],
                )

                # Get segments that overlap with our time range
                best_replacement = None
                for new_seg in segments_gen:
                    if new_seg.end < seg.start:
                        continue
                    if new_seg.start > seg.end + 1.0:
                        break

                    new_logprob = getattr(new_seg, "avg_logprob", None)
                    if new_logprob is not None and new_logprob > original_logprob:
                        if best_replacement is None or new_logprob > best_replacement.avg_logprob:
                            from transcripty.models import Word

                            words = []
                            if cfg.word_timestamps and new_seg.words:
                                words = [
                                    Word(
                                        text=w.word, start=w.start,
                                        end=w.end, probability=w.probability,
                                    )
                                    for w in new_seg.words
                                ]

                            best_replacement = Segment(
                                text=new_seg.text.strip(),
                                start=new_seg.start,
                                end=new_seg.end,
                                words=words,
                                language=info.language,
                                avg_logprob=new_logprob,
                                no_speech_prob=getattr(new_seg, "no_speech_prob", None),
                            )

                if best_replacement is not None:
                    logger.debug(
                        "Improved segment %d: logprob %.3f → %.3f, text: '%s' → '%s'",
                        idx, original_logprob, best_replacement.avg_logprob,
                        seg.text[:40], best_replacement.text[:40],
                    )
                    result.segments[idx] = best_replacement
                    improved_count += 1

            except Exception as e:
                logger.warning("Retranscription failed for segment %d: %s", idx, e)

    logger.info(
        "Quality fallback: %d/%d suspicious segments improved",
        improved_count, len(suspicious_indices),
    )
    return result
