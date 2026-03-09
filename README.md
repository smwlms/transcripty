# Transcripty

Standalone Python package for audio transcription (Whisper) and speaker diarization (Pyannote).

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
    audio_path="recording.mp3",
    model_size="small",        # tiny/base/small/medium/large-v3
    language=None,             # None = auto-detect
    word_timestamps=True,
    compute_type="int8",       # int8/float16/float32
)

print(f"Language: {result.language} ({result.language_probability:.0%})")
print(f"Duration: {result.duration:.1f}s")

for seg in result.segments:
    print(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text}")
```

### Speaker diarization

Requires the `[diarization]` extra and a HuggingFace token.

```bash
# Set token in .env or environment
echo "HF_TOKEN=hf_your_token" > .env
```

```python
from transcripty import transcribe, diarize, merge

result = transcribe("recording.mp3")

speakers = diarize(
    audio_path="recording.mp3",
    hf_token=None,             # Falls back to HF_TOKEN env var
    num_speakers=None,         # None = auto-detect
    min_speakers=1,
    max_speakers=10,
)

labeled = merge(result.segments, speakers.segments)

for seg in labeled:
    print(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.speaker}: {seg.text}")
```

## Models

All inputs and outputs use Pydantic models:

| Model                 | Fields                                             |
| --------------------- | -------------------------------------------------- |
| `Word`                | text, start, end, probability                      |
| `Segment`             | text, start, end, words                            |
| `TranscriptionResult` | segments, language, language_probability, duration |
| `DiarizationSegment`  | speaker, start, end                                |
| `DiarizationResult`   | segments, num_speakers                             |
| `LabeledSegment`      | text, start, end, speaker, words                   |

## Device detection

Automatically selects the best available compute device:

1. **CUDA** — NVIDIA GPU
2. **MPS** — Apple Silicon
3. **CPU** — fallback

## Requirements

- Python ≥ 3.10
- ffmpeg (for non-WAV audio conversion via pydub)

## License

MIT
