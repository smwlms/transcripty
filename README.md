# Transcripty

Standalone Python package for audio transcription (Whisper) and speaker diarization (Pyannote). 100% local, no cloud dependency.

## Installation

```bash
# Base — transcription only
pip install git+https://github.com/smwlms/transcripty.git

# With speaker diarization
pip install "transcripty[diarization] @ git+https://github.com/smwlms/transcripty.git"
```

## Quick start

### Transcription

```python
from transcripty import transcribe

result = transcribe(
    "recording.mp3",
    model_size="large-v3",      # see Model Comparison below
    language="nl",              # None = auto-detect
    word_timestamps=True,
)

for seg in result.segments:
    print(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text}")
```

### Accuracy-optimized transcription

For maximum accuracy on long audio, use the optimized parameters:

```python
result = transcribe(
    "meeting.mp3",
    model_size="large-v3",
    language="nl",
    vad_filter=True,                     # filter non-speech audio
    condition_on_previous_text=False,    # prevent hallucination cascades
    hallucination_silence_threshold=2.0, # skip hallucinated silence
    repetition_penalty=1.1,             # reduce repetition loops
    no_repeat_ngram_size=3,             # prevent n-gram repetitions
)
```

These parameters are auto-configured when hardware detection selects a large model (16+ GB RAM).

### Speaker diarization

Requires the `[diarization]` extra and a HuggingFace token.

```python
from transcripty import transcribe_with_speakers

segments = transcribe_with_speakers(
    "meeting.mp3",
    hf_token="hf_...",
    model_size="large-v3",
    language="nl",
    vad_filter=True,
    condition_on_previous_text=False,
)

for seg in segments:
    print(f"{seg.speaker}: {seg.text}")
```

## Available models

| Model             | Size | Best for         | Notes                             |
| ----------------- | ---- | ---------------- | --------------------------------- |
| `tiny`            | 39M  | Testing          | Fast but low quality              |
| `base`            | 74M  | Testing          | Better but still many errors      |
| `small`           | 244M | Quick drafts     | Acceptable quality                |
| `medium`          | 769M | Good balance     | Good quality, reasonable speed    |
| `large-v3`        | 1.5G | Maximum accuracy | Best WER, slowest                 |
| `large-v3-turbo`  | 809M | Best balance     | Near large-v3 accuracy, 3x faster |
| `distil-large-v3` | 756M | English only     | Hallucinates on non-English       |

## Model comparison

Benchmark on 105s Dutch audio (Apple M1 Max, int8):

| Model          | Default WER | Optimized WER | Speed |
| -------------- | ----------- | ------------- | ----- |
| large-v3       | 1.9%        | **0.8%**      | 49s   |
| large-v3-turbo | 2.7%        | **1.5%**      | 16s   |
| medium         | 5.3%        | 9.8%\*        | 29s   |
| small          | 9.5%        | 11.7%\*       | 12s   |

\*Optimized parameters improve large models but can reduce quality on smaller models. Use default parameters for tiny/base/small/medium.

WER (Word Error Rate) = percentage of incorrectly transcribed words. Lower is better.

See [`benchmarks/benchmark_report.md`](benchmarks/benchmark_report.md) for full results including long audio tests.

## Accuracy parameters

| Parameter                         | Default | Optimized | Effect                              |
| --------------------------------- | ------- | --------- | ----------------------------------- |
| `vad_filter`                      | `False` | `True`    | Silero VAD removes non-speech audio |
| `condition_on_previous_text`      | `True`  | `False`   | Prevents hallucination cascades     |
| `hallucination_silence_threshold` | `None`  | `2.0`     | Skips hallucinated silence segments |
| `repetition_penalty`              | `1.0`   | `1.1`     | Penalizes repeated tokens           |
| `no_repeat_ngram_size`            | `0`     | `3`       | Blocks exact n-gram repetitions     |

## Data models

All inputs and outputs use Pydantic models:

| Model                 | Fields                                                |
| --------------------- | ----------------------------------------------------- |
| `Word`                | text, start, end, probability                         |
| `Segment`             | text, start, end, words                               |
| `TranscriptionResult` | segments, language, language_probability, duration    |
| `DiarizationSegment`  | speaker, start, end                                   |
| `DiarizationResult`   | segments, num_speakers, embeddings                    |
| `LabeledSegment`      | text, start, end, speaker, words                      |
| `WordHighlight`       | word, start, end, probability, segment_index, speaker |

## Device detection

Automatically selects the best available compute device:

1. **CUDA** — NVIDIA GPU
2. **MPS** — Apple Silicon (used by pyannote, not by faster-whisper/CTranslate2)
3. **CPU** — fallback (used by faster-whisper on Apple Silicon)

## Documentation

- [API Reference](docs/api.md)
- [Word Highlights](docs/word-highlights.md) — frontend audio-sync integration
- [Speaker Enrollment](docs/speaker-enrollment.md) — voice profile enrollment
- [Benchmark Report](benchmarks/benchmark_report.md) — model comparison results

## Requirements

- Python >= 3.10
- ffmpeg (for non-WAV audio conversion via pydub)

## License

MIT
