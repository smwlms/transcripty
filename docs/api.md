# Transcripty API Reference

Standalone Python package for audio transcription (Whisper) and speaker diarization (pyannote).

## Installation

```bash
# Core (transcription only)
pip install git+https://github.com/smwlms/transcripty.git

# With speaker diarization
pip install "transcripty[diarization] @ git+https://github.com/smwlms/transcripty.git"
```

**Requirements:** Python 3.10+, ffmpeg installed on the system.

---

## Quick start

```python
from transcripty import transcribe, to_word_highlights

# Basic transcription
result = transcribe("recording.mp3")
print(result.language, result.duration)
for seg in result.segments:
    print(f"[{seg.start:.1f}s] {seg.text}")

# With word-level timestamps for highlighting
result = transcribe("recording.mp3", word_timestamps=True)
highlights = to_word_highlights(result.segments)
```

```python
# Full pipeline: transcription + speaker diarization
from transcripty import transcribe_with_speakers, to_srt

segments = transcribe_with_speakers("meeting.mp3", hf_token="hf_...")
print(to_srt(segments))
```

---

## Core functions

### `transcribe()`

Transcribe an audio file using faster-whisper.

```python
from transcripty import transcribe

result = transcribe(
    audio_path="recording.mp3",     # str | Path — any format (WAV, MP3, M4A, ...)
    model_size="small",             # "tiny"|"base"|"small"|"medium"|"large-v3"|"large-v3-turbo"|"distil-large-v3"
    language=None,                  # str | None — e.g. "nl", "en". None = auto-detect
    word_timestamps=True,           # bool — include word-level timing
    compute_type="int8",            # "int8" | "float16" | "float32" | "auto"
    beam_size=5,                    # int — beam search width
    prompt=None,                    # str | None — initial prompt to bias recognition
    # Accuracy & anti-hallucination parameters
    vad_filter=True,                # bool — Silero VAD filters non-speech audio
    condition_on_previous_text=False, # bool — False prevents hallucination cascades
    hallucination_silence_threshold=2.0, # float | None — skip hallucinated silence segments
    repetition_penalty=1.1,         # float — penalize repeated tokens (>1.0)
    no_repeat_ngram_size=3,         # int — prevent exact n-gram repetitions
    # Progress callback
    on_progress=lambda p, msg: print(f"{p:.0%} {msg}"),  # Callable[[float, str], None] | None
)
```

**Returns:** [`TranscriptionResult`](#transcriptionresult)

**Raises:**

- `FileNotFoundError` — audio file does not exist
- `ImportError` — faster-whisper not installed

All parameters except `audio_path` default to the current [configuration](#configuration).

**Prompt usage:** Pass domain-specific terms to improve accuracy:

```python
result = transcribe("call.mp3", prompt="TensorFlow, Kubernetes, microservices")
```

Or use the [`Vocabulary`](#vocabulary) class to manage a reusable word list.

---

### `diarize()`

Run speaker diarization on an audio file using pyannote.audio.

```python
from transcripty import diarize

speakers = diarize(
    audio_path="meeting.mp3",      # str | Path
    hf_token="hf_...",             # str | None — HuggingFace token (required)
    num_speakers=None,             # int | None — exact count. None = auto-detect
    min_speakers=1,                # int — minimum expected speakers
    max_speakers=10,               # int — maximum expected speakers
    pipeline="pyannote/speaker-diarization-3.1",  # str — pipeline model name
    on_progress=lambda p, msg: print(f"{p:.0%} {msg}"),  # Callable[[float, str], None] | None
)
```

**Returns:** [`DiarizationResult`](#diarizationresult)

**Raises:**

- `FileNotFoundError` — audio file does not exist
- `ValueError` — no HuggingFace token provided
- `ImportError` — pyannote.audio or torch not installed

**Token resolution order:** `hf_token` parameter > `configure(hf_token=...)` > `HF_TOKEN` env var.

> Requires `pip install "transcripty[diarization]"`

---

### `merge()`

Merge transcription segments with diarization to assign speaker labels.

```python
from transcripty import transcribe, diarize, merge

result = transcribe("meeting.mp3")
speakers = diarize("meeting.mp3", hf_token="hf_...")

labeled = merge(
    segments=result.segments,           # list[Segment]
    diarization=speakers.segments,      # list[DiarizationSegment]
    speaker_names=None,                 # dict[str, str] | None — e.g. {"SPEAKER_00": "Alice"}
)
```

**Returns:** `list[`[`LabeledSegment`](#labeledsegment)`]`

For each transcription segment, finds the speaker with the most temporal overlap using binary search. Unmatched segments get `UNKNOWN_SPEAKER` ("UNKNOWN").

---

### `transcribe_with_speakers()`

Convenience function that runs transcription + diarization + merge in one call.

```python
from transcripty import transcribe_with_speakers

segments = transcribe_with_speakers(
    audio_path="meeting.mp3",       # str | Path
    hf_token="hf_...",              # str | None
    num_speakers=None,              # int | None
    speaker_db=None,                # SpeakerDB | None — for speaker identification
    on_progress=lambda p, msg: print(f"{p:.0%} {msg}"),  # Callable[[float, str], None] | None
    # All transcribe() kwargs are forwarded:
    model_size="medium",
    language="nl",
    word_timestamps=True,
)
```

**Returns:** `list[`[`LabeledSegment`](#labeledsegment)`]`

When `speaker_db` is provided, speakers are identified by name instead of generic labels.

**Progress weights:** The pipeline reports weighted progress across stages:

- `0.0–0.7` — transcription
- `0.7–0.9` — diarization
- `0.9–1.0` — merge/identify

> Requires `pip install "transcripty[diarization]"`

---

## Output formatters

### `to_srt()`

Convert segments to SRT subtitle format.

```python
from transcripty import to_srt

srt_text = to_srt(segments)  # list[Segment] | list[LabeledSegment]
```

Output:

```
1
00:00:00,000 --> 00:00:02,500
Hello world

2
00:00:03,000 --> 00:00:05,000
[Alice] Second line
```

Speaker labels are included as `[Speaker]` prefix when available.

---

### `to_vtt()`

Convert segments to WebVTT subtitle format.

```python
from transcripty import to_vtt

vtt_text = to_vtt(segments)  # list[Segment] | list[LabeledSegment]
```

Output:

```
WEBVTT

00:00:00.000 --> 00:00:02.500
Hello world

00:00:03.000 --> 00:00:05.000
<v Alice>Second line
```

Speaker labels use the VTT `<v Speaker>` voice tag.

---

### `to_text()`

Convert segments to plain text.

```python
from transcripty import to_text

text = to_text(
    segments,                       # list[Segment] | list[LabeledSegment]
    include_speakers=True,          # bool — prefix with speaker name
    include_timestamps=False,       # bool — prefix with [MM:SS]
)
```

Output with speakers + timestamps:

```
[00:00] Alice: Hello world
[00:03] Bob: Hi there
```

---

### `to_word_highlights()`

Extract a flat list of word-level timing data for frontend audio-sync highlighting.

```python
from transcripty import to_word_highlights

highlights = to_word_highlights(segments)  # list[Segment] | list[LabeledSegment]
data = [h.model_dump() for h in highlights]
```

**Returns:** `list[`[`WordHighlight`](#wordhighlight)`]`

Requires `word_timestamps=True` during transcription. Segments without words are skipped.

See [Word Highlights Guide](word-highlights.md) for frontend integration with Wavesurfer.js.

---

## Data models

All models are Pydantic v2 `BaseModel` — JSON-serializable via `.model_dump()` / `.model_dump_json()`.

### `Word`

A single word with timing information.

| Field         | Type    | Default  | Description                        |
| ------------- | ------- | -------- | ---------------------------------- |
| `text`        | `str`   | required | The word text                      |
| `start`       | `float` | required | Start time in seconds              |
| `end`         | `float` | required | End time in seconds                |
| `probability` | `float` | `0.0`    | Whisper confidence score (0.0-1.0) |

---

### `Segment`

A transcription segment (phrase/sentence).

| Field   | Type         | Default  | Description                                         |
| ------- | ------------ | -------- | --------------------------------------------------- |
| `text`  | `str`        | required | Segment text                                        |
| `start` | `float`      | required | Start time in seconds                               |
| `end`   | `float`      | required | End time in seconds                                 |
| `words` | `list[Word]` | `[]`     | Word-level timestamps (when `word_timestamps=True`) |

---

### `TranscriptionResult`

Result of a `transcribe()` call.

| Field                  | Type            | Description                             |
| ---------------------- | --------------- | --------------------------------------- |
| `segments`             | `list[Segment]` | Transcribed segments                    |
| `language`             | `str`           | Detected language code (e.g. `"nl"`)    |
| `language_probability` | `float`         | Language detection confidence (0.0-1.0) |
| `duration`             | `float`         | Total audio duration in seconds         |

---

### `DiarizationSegment`

A speaker segment from diarization.

| Field     | Type    | Description                         |
| --------- | ------- | ----------------------------------- |
| `speaker` | `str`   | Speaker label (e.g. `"SPEAKER_00"`) |
| `start`   | `float` | Start time in seconds               |
| `end`     | `float` | End time in seconds                 |

---

### `DiarizationResult`

Result of a `diarize()` call.

| Field          | Type                       | Default  | Description                                  |
| -------------- | -------------------------- | -------- | -------------------------------------------- |
| `segments`     | `list[DiarizationSegment]` | required | Speaker segments                             |
| `num_speakers` | `int`                      | required | Number of detected speakers                  |
| `embeddings`   | `dict[str, list[float]]`   | `{}`     | Speaker embeddings (speaker_label -> vector) |

The `embeddings` field is populated by pyannote v4+ and used by [`SpeakerDB.identify()`](#speakerdb) for speaker recognition.

---

### `LabeledSegment`

A transcription segment with an assigned speaker label. Produced by `merge()` or `transcribe_with_speakers()`.

| Field     | Type         | Default  | Description           |
| --------- | ------------ | -------- | --------------------- |
| `text`    | `str`        | required | Segment text          |
| `start`   | `float`      | required | Start time in seconds |
| `end`     | `float`      | required | End time in seconds   |
| `speaker` | `str`        | required | Speaker label or name |
| `words`   | `list[Word]` | `[]`     | Word-level timestamps |

---

### `WordHighlight`

A single word positioned in time for frontend audio-sync highlighting. Produced by `to_word_highlights()`.

| Field           | Type          | Default  | Description                            |
| --------------- | ------------- | -------- | -------------------------------------- |
| `word`          | `str`         | required | The word text                          |
| `start`         | `float`       | required | Start time in seconds                  |
| `end`           | `float`       | required | End time in seconds                    |
| `probability`   | `float`       | `0.0`    | Whisper confidence (0.0-1.0)           |
| `segment_index` | `int`         | required | Index of the parent segment            |
| `speaker`       | `str \| None` | `None`   | Speaker label (when using diarization) |

---

### `UNKNOWN_SPEAKER`

Constant string `"UNKNOWN"`. Assigned to segments that could not be matched to any speaker during merge.

---

## Configuration

Transcripty uses a layered configuration system. Priority (highest to lowest):

1. **Programmatic** — `configure()` calls
2. **Environment variables** — `TRANSCRIPTY_*` prefix
3. **YAML config file** — `~/.transcripty/config.yaml`
4. **Hardware defaults** — auto-detected on first use

### `configure()`

Update configuration at runtime.

```python
from transcripty import configure

configure(
    model_size="medium",
    language="nl",
    word_timestamps=True,
    hf_token="hf_...",
    max_cached_models=3,
)

# Reset a field to its default
configure(language=None)    # back to auto-detect
```

**Returns:** `TranscriptyConfig`

Pass `None` explicitly to reset a nullable field. Omit a field to leave it unchanged.

---

### `get_config()`

Get the current configuration singleton.

```python
from transcripty import get_config

cfg = get_config()
print(cfg.model_size)       # "medium"
print(cfg.compute_type)     # "int8"
print(cfg.language)         # None (auto-detect)
```

**Returns:** `TranscriptyConfig`

---

### Config fields

| Field               | Type          | Default   | Env var                         | Description                |
| ------------------- | ------------- | --------- | ------------------------------- | -------------------------- |
| `model_size`        | `str`         | `"small"` | `TRANSCRIPTY_MODEL_SIZE`        | Whisper model size         |
| `compute_type`      | `str`         | `"int8"`  | `TRANSCRIPTY_COMPUTE_TYPE`      | Quantization type          |
| `language`          | `str \| None` | `None`    | `TRANSCRIPTY_LANGUAGE`          | Language code, None = auto |
| `beam_size`         | `int`         | `5`       | `TRANSCRIPTY_BEAM_SIZE`         | Beam search width          |
| `word_timestamps`   | `bool`        | `True`    | `TRANSCRIPTY_WORD_TIMESTAMPS`   | Word-level timing          |
| `hf_token`          | `str \| None` | `None`    | `TRANSCRIPTY_HF_TOKEN`          | HuggingFace API token      |
| `num_speakers`      | `int \| None` | `None`    | `TRANSCRIPTY_NUM_SPEAKERS`      | Exact speaker count        |
| `min_speakers`      | `int`         | `1`       | `TRANSCRIPTY_MIN_SPEAKERS`      | Minimum speakers           |
| `max_speakers`      | `int`         | `10`      | `TRANSCRIPTY_MAX_SPEAKERS`      | Maximum speakers           |
| `vocabulary_path`   | `str \| None` | `None`    | `TRANSCRIPTY_VOCABULARY_PATH`   | Path to vocabulary JSON    |
| `speaker_db_path`   | `str \| None` | `None`    | `TRANSCRIPTY_SPEAKER_DB_PATH`   | Path to speaker DB JSON    |
| `max_cached_models` | `int`         | `2`       | `TRANSCRIPTY_MAX_CACHED_MODELS` | Max models in cache (>=1)  |
| `num_workers`       | `int`         | `1`       | `TRANSCRIPTY_NUM_WORKERS`       | Worker count (>=1)         |

**Accuracy & anti-hallucination fields:**

| Field                             | Type            | Default | Env var                                       | Description                    |
| --------------------------------- | --------------- | ------- | --------------------------------------------- | ------------------------------ |
| `vad_filter`                      | `bool`          | `False` | `TRANSCRIPTY_VAD_FILTER`                      | Silero VAD filters non-speech  |
| `condition_on_previous_text`      | `bool`          | `True`  | `TRANSCRIPTY_CONDITION_ON_PREVIOUS_TEXT`      | Use previous output as context |
| `hallucination_silence_threshold` | `float \| None` | `None`  | `TRANSCRIPTY_HALLUCINATION_SILENCE_THRESHOLD` | Skip hallucinated silence      |
| `repetition_penalty`              | `float`         | `1.0`   | `TRANSCRIPTY_REPETITION_PENALTY`              | Penalize repeated tokens       |
| `no_repeat_ngram_size`            | `int`           | `0`     | `TRANSCRIPTY_NO_REPEAT_NGRAM_SIZE`            | Block n-gram repetitions       |
| `cpu_threads`                     | `int`           | `0`     | `TRANSCRIPTY_CPU_THREADS`                     | CTranslate2 threads (0=auto)   |

> **Note:** On machines with 16+ GB RAM, hardware detection automatically sets optimized defaults for large models (vad_filter=True, condition_on_previous_text=False, etc.). These parameters improve accuracy for large models but can reduce quality on smaller models.

### YAML config file

Create `~/.transcripty/config.yaml`:

```yaml
model_size: large-v3
compute_type: int8
language: nl
beam_size: 5
word_timestamps: true
min_speakers: 1
max_speakers: 6
max_cached_models: 3

# Accuracy settings (recommended for large models)
vad_filter: true
condition_on_previous_text: false
hallucination_silence_threshold: 2.0
repetition_penalty: 1.1
no_repeat_ngram_size: 3
```

Requires `pydantic-settings[yaml]` for YAML support.

---

## Speaker identification

### `SpeakerDB`

Database of known speaker voice profiles. Enroll speakers from reference audio, then identify them in diarization results.

```python
from transcripty import SpeakerDB

db = SpeakerDB()
```

> Requires `pip install "transcripty[diarization]"`

#### `db.enroll(name, audio_path, hf_token=None)`

Enroll a speaker by extracting their voice embedding from reference audio. The audio should contain only the target speaker's voice.

```python
db.enroll("Alice", "alice_sample.wav", hf_token="hf_...")
db.enroll("Bob", "bob_sample.wav")
```

#### `db.enroll_from_embedding(name, embedding)`

Enroll a speaker directly from a pre-computed embedding vector.

```python
db.enroll_from_embedding("Alice", [0.12, -0.34, ...])
```

#### `db.identify(result, threshold=0.5)`

Identify speakers in a diarization result using exclusive greedy matching. Each speaker label maps to at most one profile, and vice versa. Highest-scoring pairs are matched first.

```python
speakers = diarize("meeting.mp3", hf_token="hf_...")
names = db.identify(speakers, threshold=0.5)
# {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
```

**Parameters:**

- `result` — `DiarizationResult` with embeddings
- `threshold` — minimum cosine similarity to consider a match (0.0-1.0)

**Returns:** `dict[str, str]` — speaker label to name mapping

#### `db.save(path)` / `SpeakerDB.load(path)`

Persist and restore speaker profiles as JSON.

```python
db.save("speakers.json")
db = SpeakerDB.load("speakers.json")
```

#### Properties

- `db.names` — `list[str]` of enrolled speaker names
- `len(db)` — number of enrolled speakers

---

## Vocabulary

### `Vocabulary`

Manage a list of domain-specific words to improve Whisper transcription accuracy.

```python
from transcripty import Vocabulary

vocab = Vocabulary(["TensorFlow", "Kubernetes", "microservices"])
vocab.add("PyTorch")
vocab.remove("microservices")

# Use as Whisper initial_prompt
result = transcribe("call.mp3", prompt=vocab.as_prompt())
```

#### `vocab.as_prompt()`

Generate a comma-separated prompt string: `"TensorFlow, Kubernetes, PyTorch"`.

#### `vocab.save(path)` / `Vocabulary.load(path)`

Persist and restore as JSON.

```python
vocab.save("vocabulary.json")
vocab = Vocabulary.load("vocabulary.json")
```

#### Properties

- `len(vocab)` — number of words
- `vocab.words` — the word list

---

## Hardware detection

### `detect_hardware()`

Detect hardware capabilities and return a profile. Result is cached.

```python
from transcripty.hardware import detect_hardware

profile = detect_hardware()
print(profile.cpu)        # "Apple M2 Pro"
print(profile.ram_gb)     # 16.0
print(profile.device)     # "mps"
print(profile.gpu)        # None (Apple Silicon uses MPS, not discrete GPU)
```

**Returns:** `HardwareProfile`

| Field    | Type          | Description                               |
| -------- | ------------- | ----------------------------------------- |
| `cpu`    | `str`         | CPU model name                            |
| `cores`  | `int`         | Number of CPU cores                       |
| `ram_gb` | `float`       | RAM in gigabytes                          |
| `gpu`    | `str \| None` | GPU name (NVIDIA only)                    |
| `mps`    | `bool`        | Apple Silicon MPS available               |
| `device` | `str`         | Best device: `"cuda"` / `"mps"` / `"cpu"` |

#### `profile.suggest_settings()`

Get optimal Transcripty settings for the detected hardware.

```python
suggestions = profile.suggest_settings()
# {"model_size": "medium", "compute_type": "float16", ...}
```

---

## Cache management

### `clear_cache()`

Clear all model and pipeline caches. Useful to free GPU/CPU memory.

```python
from transcripty import clear_cache

clear_cache()   # clears both Whisper models and pyannote pipelines
```

### Cache behavior

- Models and pipelines are cached in memory for reuse across calls
- Maximum `max_cached_models` entries per cache (default: 2)
- Eviction: FIFO (oldest inserted entry is removed first)
- Thread-safe: all cache operations are protected with `threading.Lock`

---

## CLI

Entry point: `transcripty`

### `transcripty hardware`

Show detected hardware and recommended settings.

```
$ transcripty hardware
CPU:      Apple M2 Pro
Cores:    10
RAM:      16.0 GB
GPU:      None
MPS:      True
Device:   mps

Recommended settings:
  model_size: medium
  compute_type: float16
```

### `transcripty config`

Show current configuration values.

```
$ transcripty config
model_size: small
compute_type: int8
language: None
...
```

### `transcripty run`

Transcribe an audio file.

```bash
# Basic transcription
transcripty run recording.mp3

# With options
transcripty run recording.mp3 \
  --model medium \
  --language nl \
  --compute-type float16 \
  --format srt \
  --output transcript.srt

# With speaker diarization
transcripty run meeting.mp3 --diarize --format srt
```

| Option                       | Default        | Description                             |
| ---------------------------- | -------------- | --------------------------------------- |
| `--model`                    | config         | Whisper model size                      |
| `--language`                 | auto           | Language code                           |
| `--compute-type`             | config         | `int8` / `float16` / `float32` / `auto` |
| `--format`                   | `text`         | Output format: `text` / `srt` / `vtt`   |
| `--diarize` / `--no-diarize` | `--no-diarize` | Enable speaker diarization              |
| `--vad` / `--no-vad`         | config         | Enable/disable VAD filter               |
| `--output`, `-o`             | stdout         | Output file path                        |

### `transcripty benchmark`

Benchmark transcription speed across models.

```bash
transcripty benchmark recording.mp3 --models "small,medium,large-v3" -o results.json
```

| Option           | Default        | Description                 |
| ---------------- | -------------- | --------------------------- |
| `--models`       | `small,medium` | Comma-separated model sizes |
| `--language`     | auto           | Language code               |
| `--output`, `-o` | none           | Save results as JSON        |

### Global options

| Option            | Description          |
| ----------------- | -------------------- |
| `--verbose`, `-v` | Enable debug logging |

---

## FastAPI integration example

```python
from fastapi import FastAPI, UploadFile
from transcripty import (
    configure,
    transcribe,
    transcribe_with_speakers,
    to_word_highlights,
    TranscriptionResult,
    LabeledSegment,
    WordHighlight,
)

app = FastAPI()

# Configure once at startup
configure(
    model_size="large-v3",
    compute_type="int8",
    word_timestamps=True,
    vad_filter=True,
    condition_on_previous_text=False,
    hallucination_silence_threshold=2.0,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
)


@app.post("/transcribe", response_model=TranscriptionResult)
async def api_transcribe(audio_path: str, language: str | None = None):
    def on_progress(progress: float, message: str):
        # e.g. push to WebSocket, update DB, log
        logger.info("Progress: %.0f%% — %s", progress * 100, message)

    return transcribe(audio_path, language=language, on_progress=on_progress)


@app.post("/transcribe/speakers", response_model=list[LabeledSegment])
async def api_transcribe_speakers(audio_path: str, hf_token: str):
    return transcribe_with_speakers(audio_path, hf_token=hf_token)


@app.post("/transcribe/highlights", response_model=list[WordHighlight])
async def api_highlights(audio_path: str):
    result = transcribe(audio_path)
    return to_word_highlights(result.segments)
```

---

## Error handling

| Exception           | Raised by                                 | When                                  |
| ------------------- | ----------------------------------------- | ------------------------------------- |
| `FileNotFoundError` | `transcribe()`, `diarize()`               | Audio file does not exist             |
| `ValueError`        | `diarize()`                               | No HuggingFace token available        |
| `ImportError`       | `transcribe()`                            | faster-whisper not installed          |
| `ImportError`       | `diarize()`, `transcribe_with_speakers()` | pyannote.audio or torch not installed |

---

## Thread safety

Transcripty is designed for use in async servers (FastAPI, etc.):

- **Model cache** — `threading.Lock` protects Whisper model loading/eviction
- **Pipeline cache** — `threading.Lock` protects pyannote pipeline loading/eviction
- **Config singleton** — `threading.Lock` protects reads and writes
- **Eviction** — FIFO, max `max_cached_models` entries per cache

Multiple threads can safely call `transcribe()` and `diarize()` concurrently.
