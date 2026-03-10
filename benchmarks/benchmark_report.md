# Transcripty Benchmark Report

**Date:** 2026-03-10 14:46
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
| tiny | 5.69s | 0.054x | **40.9%** | 108/264 | 287 |
| base | 6.42s | 0.061x | **24.6%** | 65/264 | 270 |
| small | 17.24s | 0.164x | **9.5%** | 25/264 | 265 |
| medium | 44.9s | 0.426x | **5.3%** | 14/264 | 265 |
| large-v3 | 72.72s | 0.69x | **1.9%** | 5/264 | 265 |
| large-v3-turbo | 25.84s | 0.245x | **2.7%** | 7/264 | 266 |
| distil-large-v3 | 18.23s | 0.173x | **96.2%** | 254/264 | 185 |

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
| tiny | 4.26s | 0.04x | **44.7%** | 118/264 | 270 |
| base | 5.81s | 0.055x | **26.1%** | 69/264 | 271 |
| small | 11.92s | 0.113x | **11.7%** | 31/264 | 262 |
| medium | 27.99s | 0.265x | **9.8%** | 26/264 | 264 |
| large-v3 | 54.73s | 0.519x | **0.8%** | 2/264 | 263 |
| large-v3-turbo | 16.52s | 0.157x | **1.5%** | 4/264 | 263 |
| distil-large-v3 | 17.95s | 0.17x | **94.3%** | 249/264 | 252 |

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

## 3. Long Audio Benchmark (57 min meeting, 4 speakers)

Diarization (shared): 205.4s (4 speakers detected)

### Preset: `default`

| Model | Transcription | Total | RT Factor | WER | Speaker ID |
|-------|--------------|------|-----------|-----|------------|
| small | 554.44s | 759.87s | 0.162x | **42.7%** | Yes |
| medium | 1181.99s | 1387.42s | 0.344x | **46.7%** | Yes |
| large-v3 | 4311.35s | 4516.78s | 1.257x | **37.0%** | Yes |
| large-v3-turbo | 777.92s | 983.35s | 0.227x | **39.0%** | Yes |

### Preset: `optimized`

| Model | Transcription | Total | RT Factor | WER | Speaker ID |
|-------|--------------|------|-----------|-----|------------|
| small | 394.0s | 599.44s | 0.115x | **42.7%** | Yes |
| medium | 944.46s | 1149.89s | 0.275x | **36.3%** | Yes |
| large-v3 | 1633.74s | 1839.17s | 0.476x | **31.3%** | Yes |
| large-v3-turbo | 532.53s | 737.97s | 0.155x | **34.0%** | Yes |

### WER improvement on long audio

| Model | Default WER | Optimized WER | Improvement |
|-------|-------------|---------------|-------------|
| small | 42.7% | 42.7% | +0.0pp |
| medium | 46.7% | 36.3% | +10.4pp |
| large-v3 | 37.0% | 31.3% | +5.7pp |
| large-v3-turbo | 39.0% | 34.0% | +5.0pp |

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
