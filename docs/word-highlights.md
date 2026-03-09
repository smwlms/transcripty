# Word Highlights API

Transcripty can produce word-level timing data for synchronized text highlighting during audio playback. This is useful for karaoke-style UIs, accessibility features, or any application where the current word should be visually highlighted as audio plays.

## Quick start

```python
from transcripty import transcribe, to_word_highlights

# Transcribe with word timestamps enabled
result = transcribe("recording.mp3", word_timestamps=True)

# Extract flat word list for frontend consumption
highlights = to_word_highlights(result.segments)

# Serialize to JSON (e.g. for a REST API response)
data = [h.model_dump() for h in highlights]
```

## Output format

`to_word_highlights()` returns a flat `list[WordHighlight]` — one entry per word, ordered chronologically:

```json
[
  {
    "word": "Hello",
    "start": 0.0,
    "end": 0.45,
    "probability": 0.95,
    "segment_index": 0,
    "speaker": null
  },
  {
    "word": "world",
    "start": 0.45,
    "end": 0.9,
    "probability": 0.92,
    "segment_index": 0,
    "speaker": null
  },
  {
    "word": "How",
    "start": 1.1,
    "end": 1.3,
    "probability": 0.88,
    "segment_index": 1,
    "speaker": null
  }
]
```

### Fields

| Field           | Type          | Description                                        |
| --------------- | ------------- | -------------------------------------------------- |
| `word`          | `str`         | The word text                                      |
| `start`         | `float`       | Start time in seconds                              |
| `end`           | `float`       | End time in seconds                                |
| `probability`   | `float`       | Whisper's confidence (0.0-1.0). Default `0.0`      |
| `segment_index` | `int`         | Index of the parent segment (for grouping/context) |
| `speaker`       | `str \| null` | Speaker label (only when using diarization)        |

## With speaker diarization

When combined with `transcribe_with_speakers()`, each word carries the speaker label:

```python
from transcripty import transcribe_with_speakers, to_word_highlights

segments = transcribe_with_speakers(
    "meeting.mp3",
    hf_token="hf_...",
    word_timestamps=True,
)

highlights = to_word_highlights(segments)
# Each highlight now has speaker="SPEAKER_00", "SPEAKER_01", etc.
```

## FastAPI endpoint example

```python
from fastapi import FastAPI
from transcripty import transcribe, to_word_highlights, WordHighlight

app = FastAPI()

@app.post("/transcribe", response_model=list[WordHighlight])
async def transcribe_audio(audio_path: str):
    result = transcribe(audio_path, word_timestamps=True)
    return to_word_highlights(result.segments)
```

## Frontend integration (Wavesurfer.js)

[Wavesurfer.js](https://wavesurfer.xyz/) is a popular audio player with waveform visualization. Here's how to integrate word highlighting:

### Install

```bash
npm install wavesurfer.js
```

### Implementation

```typescript
import WaveSurfer from "wavesurfer.js";

interface WordHighlight {
  word: string;
  start: number;
  end: number;
  probability: number;
  segment_index: number;
  speaker: string | null;
}

// 1. Create player
const wavesurfer = WaveSurfer.create({
  container: "#waveform",
  waveColor: "#ddd",
  progressColor: "#2196F3",
});

wavesurfer.load("/audio/recording.mp3");

// 2. Fetch highlights from your API
const highlights: WordHighlight[] = await fetch("/api/transcribe").then((r) =>
  r.json(),
);

// 3. Render words as clickable spans
const container = document.getElementById("transcript")!;
let currentSegment = -1;

highlights.forEach((h, i) => {
  // Add line break between segments
  if (h.segment_index !== currentSegment) {
    if (currentSegment !== -1)
      container.appendChild(document.createElement("br"));
    currentSegment = h.segment_index;

    // Optional: show speaker label
    if (h.speaker) {
      const label = document.createElement("strong");
      label.textContent = `${h.speaker}: `;
      container.appendChild(label);
    }
  }

  const span = document.createElement("span");
  span.textContent = h.word + " ";
  span.dataset.index = String(i);
  span.style.cursor = "pointer";
  span.style.transition = "background-color 0.1s";

  // Click to seek
  span.addEventListener("click", () => {
    wavesurfer.seekTo(h.start / wavesurfer.getDuration());
  });

  container.appendChild(span);
});

// 4. Highlight current word during playback
const wordSpans =
  container.querySelectorAll<HTMLSpanElement>("span[data-index]");

wavesurfer.on("timeupdate", (currentTime: number) => {
  wordSpans.forEach((span, i) => {
    const h = highlights[i];
    const isActive = currentTime >= h.start && currentTime < h.end;
    span.style.backgroundColor = isActive ? "#FFEB3B" : "transparent";
    span.style.fontWeight = isActive ? "bold" : "normal";
  });
});
```

### HTML

```html
<div id="waveform"></div>
<div id="transcript" style="font-size: 1.1rem; line-height: 1.8;"></div>
```

## Filtering low-confidence words

The `probability` field lets you filter or style words based on Whisper's confidence:

```typescript
// Frontend: dim low-confidence words
wordSpans.forEach((span, i) => {
  if (highlights[i].probability < 0.5) {
    span.style.opacity = "0.5";
  }
});
```

```python
# Backend: filter before sending to frontend
highlights = to_word_highlights(result.segments)
confident = [h for h in highlights if h.probability >= 0.5]
```
