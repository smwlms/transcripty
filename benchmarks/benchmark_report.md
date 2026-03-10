# Transcripty Benchmark Report

**Date:** 2026-03-10 11:42
**Hardware:** Apple M1 Max, 10 cores, 32 GB RAM
**Compute type:** int8 (CTranslate2 on CPU)
**Reference:** Plaud cloud transcription

---

## 1. Speaker Enrollment

- **Speaker:** Samuel Willems
- **Audio:** 105s NL read-aloud text
- **Diarization time:** 0s
- **Embedding dimension:** 256

---

## 2. Model Comparison — Short Audio (105s)

All models tested with two parameter presets:
- **default:** vanilla faster-whisper settings
- **optimized:** VAD filter, no condition on previous text, hallucination threshold, repetition penalty

### Preset: `default`

| Model | Time | RT Factor | WER | Errors | Words |
|-------|------|-----------|-----|--------|-------|
| tiny | 5.46s | 0.052x | **40.9%** | 108/264 | 287 |
| base | 6.51s | 0.062x | **24.6%** | 65/264 | 270 |
| small | 15.71s | 0.149x | **9.5%** | 25/264 | 265 |
| medium | 46.64s | 0.442x | **5.3%** | 14/264 | 265 |
| large-v3 | 74.72s | 0.709x | **1.9%** | 5/264 | 265 |
| large-v3-turbo | 71.5s | 0.678x | **2.7%** | 7/264 | 266 |
| distil-large-v3 | 18.75s | 0.178x | **96.2%** | 254/264 | 185 |

### Preset: `optimized`

```
vad_filter=True
condition_on_previous_text=False
hallucination_silence_threshold=2.0
repetition_penalty=1.1
no_repeat_ngram_size=3
```

| Model | Time | RT Factor | WER | Errors | Words |
|-------|------|-----------|-----|--------|-------|
| tiny | 3.71s | 0.035x | **44.7%** | 118/264 | 270 |
| base | 5.68s | 0.054x | **26.1%** | 69/264 | 271 |
| small | 11.59s | 0.11x | **11.7%** | 31/264 | 262 |
| medium | 29.25s | 0.277x | **9.8%** | 26/264 | 264 |
| large-v3 | 49.22s | 0.467x | **0.8%** | 2/264 | 263 |
| large-v3-turbo | 16.26s | 0.154x | **1.5%** | 4/264 | 263 |
| distil-large-v3 | 18.34s | 0.174x | **94.3%** | 249/264 | 252 |

### Best models

- **Best default:** `large-v3` (WER 1.9%)
- **Best optimized:** `large-v3` (WER 0.8%)

### WER improvement (default → optimized)

| Model | Default WER | Optimized WER | Improvement |
|-------|-------------|---------------|-------------|
| tiny | 40.9% | 44.7% | -3.8pp |
| base | 24.6% | 26.1% | -1.5pp |
| small | 9.5% | 11.7% | -2.2pp |
| medium | 5.3% | 9.8% | -4.5pp |
| large-v3 | 1.9% | 0.8% | +1.1pp |
| large-v3-turbo | 2.7% | 1.5% | +1.2pp |
| distil-large-v3 | 96.2% | 94.3% | +1.9pp |

---

## 4. Recommendations

### Choosing a model

| Use case | Model | Preset | Expected WER |
|----------|-------|--------|--------------|
| Maximum accuracy | large-v3 | optimized | Best |
| Good balance | large-v3-turbo | optimized | Very good |
| Fast processing | medium | optimized | Good |
| Quick draft | small | optimized | Acceptable |

### Optimized parameters (recommended)

```python
from transcripty import transcribe

result = transcribe(
    "audio.mp3",
    model_size="large-v3",
    language="nl",
    vad_filter=True,
    condition_on_previous_text=False,
    hallucination_silence_threshold=2.0,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
)
```

### Parameter explanation

- **vad_filter:** Silero VAD filters non-speech audio, reducing hallucinations
- **condition_on_previous_text:** When False, prevents hallucination cascades on long audio
- **hallucination_silence_threshold:** Skips segments generated after silence periods
- **repetition_penalty:** Penalizes repeated tokens (>1.0 reduces loops)
- **no_repeat_ngram_size:** Prevents exact n-gram repetitions

### What is WER?

Word Error Rate (WER) measures transcription accuracy. It counts substitutions (wrong words), insertions (extra words), and deletions (missing words) divided by the total reference words. Lower is better — 0% is perfect.
