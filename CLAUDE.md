# Transcripty — Standalone audio transcription & diarization package

## Quick context

Python package dat Whisper transcriptie en Pyannote speaker diarization aanbiedt als clean, herbruikbare library. Geëxtraheerd uit TranscriberApp. Primaire consumer: Plaude (Plaud audio sync tool).

## Tech stack

- Python 3.10+
- faster-whisper (CTranslate2-based Whisper)
- pyannote.audio (optioneel, voor speaker diarization)
- pydantic v2 (input/output models)
- pydub + ffmpeg (audio conversie)
- pytest + ruff (dev tooling)
- hatchling (build backend)

## Commands

```bash
pip install -e ".[dev]"           # Install dev dependencies
pip install -e ".[diarization]"   # Install with diarization support
pytest                             # Run tests
ruff check .                       # Lint
ruff format .                      # Format
```

## Belangrijke paden

- `transcripty/` — Package root
  - `models.py` — Pydantic models (Word, Segment, TranscriptionResult, etc.)
  - `transcribe.py` — Whisper wrapper, `transcribe()` functie
  - `diarize.py` — Pyannote wrapper, `diarize()` functie
  - `merge.py` — Merge transcriptie + diarization
  - `device.py` — CUDA/MPS/CPU detectie
  - `audio.py` — WAV conversie context manager
  - `__init__.py` — Public API (lazy import voor diarize)
- `tests/` — pytest tests
- `pyproject.toml` — Package config, dependencies, optionals

## Business rules

- Diarization is optioneel — base install werkt zonder torch/pyannote
- HF token: functie parameter > HF_TOKEN env var > .env file
- Audio conversie naar WAV is automatisch (pydub)
- Device detectie: CUDA > MPS > CPU, gecached na eerste call
- `diarize` wordt lazy geïmporteerd om torch import te vermijden bij base install

## Agent hints

- **data-engineer**: Geen database in dit project, pure library
- **reviewer**: Let op lazy import pattern in **init**.py, check type hints
- **qa**: Tests draaien zonder GPU/models via mocks
- **pm**: Package is dependency voor Plaude project
