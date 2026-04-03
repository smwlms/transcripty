"""Prosodic feature extraction for speaker discrimination.

Extracts shimmer, jitter, pitch statistics, and speaking rate —
features that help distinguish speakers with similar timbres (e.g. twins).

Usage:
    from transcripty.prosody import extract_prosodic_features
    features = extract_prosodic_features("audio.mp3")
    # Returns ProsodicFeatures with normalized feature vector
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ProsodicFeatures(BaseModel):
    """Prosodic feature vector for a speaker."""

    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_range: float = 0.0
    jitter: float = 0.0  # pitch perturbation (cycle-to-cycle variation)
    shimmer: float = 0.0  # amplitude perturbation
    speaking_rate: float = 0.0  # syllables per second proxy
    energy_mean: float = 0.0
    energy_std: float = 0.0
    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0
    mfcc_delta_std: float = 0.0  # articulation dynamics
    hnr: float = 0.0  # harmonics-to-noise ratio

    def as_vector(self) -> list[float]:
        """Return normalized feature vector for similarity computation."""
        return [
            self.pitch_mean,
            self.pitch_std,
            self.pitch_range,
            self.jitter,
            self.shimmer,
            self.speaking_rate,
            self.energy_mean,
            self.energy_std,
            self.spectral_centroid_mean,
            self.spectral_centroid_std,
            self.mfcc_delta_std,
            self.hnr,
        ]

    @staticmethod
    def feature_dim() -> int:
        return 12


def _compute_jitter(f0: np.ndarray) -> float:
    """Compute jitter (relative average perturbation) from F0 contour."""
    voiced = f0[f0 > 0]
    if len(voiced) < 3:
        return 0.0

    periods = 1.0 / voiced
    diffs = np.abs(np.diff(periods))
    return float(np.mean(diffs) / np.mean(periods)) if np.mean(periods) > 0 else 0.0


def _compute_shimmer(y: np.ndarray, sr: int, f0: np.ndarray, hop_length: int) -> float:
    """Compute shimmer (amplitude perturbation) from audio and F0."""
    voiced_frames = np.where(f0 > 0)[0]
    if len(voiced_frames) < 3:
        return 0.0

    # Get amplitude at each voiced frame
    amplitudes = []
    for frame_idx in voiced_frames:
        start = frame_idx * hop_length
        end = min(start + hop_length * 2, len(y))
        if end > start:
            amplitudes.append(np.sqrt(np.mean(y[start:end] ** 2)))

    if len(amplitudes) < 3:
        return 0.0

    amplitudes = np.array(amplitudes)
    diffs = np.abs(np.diff(amplitudes))
    mean_amp = np.mean(amplitudes)
    return float(np.mean(diffs) / mean_amp) if mean_amp > 0 else 0.0


def _compute_hnr(y: np.ndarray, sr: int) -> float:
    """Estimate Harmonics-to-Noise Ratio using autocorrelation."""
    # Simple HNR estimate via autocorrelation
    frame_len = min(len(y), sr)  # 1 second max
    y_frame = y[:frame_len].astype(np.float64)

    # Autocorrelation
    corr = np.correlate(y_frame, y_frame, mode="full")
    corr = corr[len(corr) // 2:]

    if len(corr) < 2 or corr[0] == 0:
        return 0.0

    corr = corr / corr[0]

    # Find first peak after zero crossing (fundamental period)
    min_lag = int(sr / 500)  # max 500 Hz
    max_lag = int(sr / 60)  # min 60 Hz

    if max_lag >= len(corr):
        max_lag = len(corr) - 1

    search = corr[min_lag:max_lag]
    if len(search) == 0:
        return 0.0

    peak_val = float(np.max(search))
    if peak_val <= 0:
        return 0.0

    # HNR in dB
    hnr = 10 * np.log10(peak_val / (1 - peak_val)) if peak_val < 1 else 30.0
    return float(np.clip(hnr, 0, 40))


def extract_prosodic_features(
    audio_path: str | Path,
    start_s: float | None = None,
    end_s: float | None = None,
) -> ProsodicFeatures:
    """Extract prosodic features from audio.

    Args:
        audio_path: Path to audio file.
        start_s: Optional start time in seconds.
        end_s: Optional end time in seconds.

    Returns:
        ProsodicFeatures with 12-dimensional feature vector.
    """
    import librosa

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    # Trim to segment
    if start_s is not None or end_s is not None:
        start_sample = int((start_s or 0) * sr)
        end_sample = int((end_s or len(y) / sr) * sr)
        end_sample = min(end_sample, len(y))
        y = y[start_sample:end_sample]

    if len(y) < sr * 0.5:  # less than 0.5 seconds
        logger.warning("Audio too short for prosodic analysis (%.2fs)", len(y) / sr)
        return ProsodicFeatures()

    hop_length = 512

    # F0 (pitch) extraction
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=60, fmax=500, sr=sr, hop_length=hop_length,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced_f0 = f0[f0 > 0]

    pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else 0.0
    pitch_range = float(np.ptp(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    # Jitter and shimmer
    jitter = _compute_jitter(f0)
    shimmer = _compute_shimmer(y, sr, f0, hop_length)

    # Speaking rate proxy (number of onsets per second)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    duration_s = len(y) / sr
    speaking_rate = len(onsets) / duration_s if duration_s > 0 else 0.0

    # Energy (RMS)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    # Spectral centroid (brightness of voice)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))

    # MFCC delta (articulation dynamics)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_std = float(np.mean(np.std(mfcc_delta, axis=1)))

    # HNR
    hnr = _compute_hnr(y, sr)

    features = ProsodicFeatures(
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        pitch_range=pitch_range,
        jitter=jitter,
        shimmer=shimmer,
        speaking_rate=speaking_rate,
        energy_mean=energy_mean,
        energy_std=energy_std,
        spectral_centroid_mean=centroid_mean,
        spectral_centroid_std=centroid_std,
        mfcc_delta_std=mfcc_delta_std,
        hnr=hnr,
    )

    logger.debug(
        "Prosodic features: pitch=%.0fHz, jitter=%.4f, shimmer=%.4f, rate=%.1f/s, HNR=%.1fdB",
        pitch_mean, jitter, shimmer, speaking_rate, hnr,
    )

    return features


def extract_prosodic_for_segments(
    audio_path: str | Path,
    speaker_segments: dict[str, list[tuple[float, float]]],
) -> dict[str, ProsodicFeatures]:
    """Extract prosodic features for each speaker from their segments.

    Concatenates segment audio per speaker and extracts features from
    the combined signal. Uses up to 60 seconds per speaker.

    Args:
        audio_path: Path to audio file.
        speaker_segments: Dict mapping speaker labels to (start, end) tuples.

    Returns:
        Dict mapping speaker labels to ProsodicFeatures.
    """
    import librosa

    audio_path = Path(audio_path)
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    results = {}
    max_samples = 60 * sr

    for speaker_label, segments in speaker_segments.items():
        chunks = []
        total_samples = 0

        for start, end in segments:
            if total_samples >= max_samples:
                break
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            end_sample = min(end_sample, len(y))
            chunk = y[start_sample:end_sample]
            chunks.append(chunk)
            total_samples += len(chunk)

        if not chunks or total_samples < sr * 0.5:
            results[speaker_label] = ProsodicFeatures()
            continue

        combined = np.concatenate(chunks)
        if len(combined) > max_samples:
            combined = combined[:max_samples]

        # Extract features from combined audio
        # We write to a temp file since extract_prosodic_features expects a path
        # Instead, inline the extraction on the numpy array
        hop_length = 512

        f0, _, _ = librosa.pyin(combined, fmin=60, fmax=500, sr=sr, hop_length=hop_length)
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced_f0 = f0[f0 > 0]

        pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else 0.0
        pitch_range = float(np.ptp(voiced_f0)) if len(voiced_f0) > 0 else 0.0

        jitter = _compute_jitter(f0)
        shimmer = _compute_shimmer(combined, sr, f0, hop_length)

        onset_env = librosa.onset.onset_strength(y=combined, sr=sr, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length,
        )
        duration_s = len(combined) / sr
        speaking_rate = len(onsets) / duration_s if duration_s > 0 else 0.0

        rms = librosa.feature.rms(y=combined, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(y=combined, sr=sr, hop_length=hop_length)[0]

        mfcc = librosa.feature.mfcc(y=combined, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)

        hnr = _compute_hnr(combined, sr)

        results[speaker_label] = ProsodicFeatures(
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            pitch_range=pitch_range,
            jitter=jitter,
            shimmer=shimmer,
            speaking_rate=speaking_rate,
            energy_mean=float(np.mean(rms)),
            energy_std=float(np.std(rms)),
            spectral_centroid_mean=float(np.mean(centroid)),
            spectral_centroid_std=float(np.std(centroid)),
            mfcc_delta_std=float(np.mean(np.std(mfcc_delta, axis=1))),
            hnr=hnr,
        )

        logger.debug(
            "Prosodic for %s: pitch=%.0fHz, jitter=%.4f, shimmer=%.4f (%.1fs audio)",
            speaker_label, pitch_mean, jitter, shimmer, duration_s,
        )

    return results


def prosodic_similarity(a: ProsodicFeatures, b: ProsodicFeatures) -> float:
    """Compute similarity between two prosodic feature vectors.

    Uses cosine similarity on z-score normalized features.
    Returns value between 0 and 1 (higher = more similar).
    """
    va = np.array(a.as_vector())
    vb = np.array(b.as_vector())

    # Skip if either is all zeros
    if np.all(va == 0) or np.all(vb == 0):
        return 0.0

    # Cosine similarity
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    cosine = float(np.dot(va, vb) / (norm_a * norm_b))
    # Clamp to [0, 1]
    return max(0.0, min(1.0, cosine))
