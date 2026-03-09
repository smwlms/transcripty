# Transcripty — Standalone audio transcription & diarization package

## Quick context

Python package dat Whisper transcriptie en Pyannote speaker diarization aanbiedt als clean, herbruikbare library. Geëxtraheerd uit TranscriberApp. Primaire consumer: Plaude (FastAPI async server).

## Tech stack

- Python 3.10+
- faster-whisper (CTranslate2-based Whisper)
- pyannote.audio (optioneel, voor speaker diarization)
- pydantic v2 + pydantic-settings (models & config)
- pydub + ffmpeg (audio conversie)
- click (CLI)
- pytest + ruff (dev tooling)
- hatchling (build backend)

## Commands

```bash
pip install -e ".[dev]"           # Install dev dependencies
pip install -e ".[diarization]"   # Install with diarization support
pytest                             # Run tests (97 tests)
ruff check .                       # Lint
ruff format .                      # Format
transcripty hardware               # Show detected hardware
transcripty config                 # Show current config
transcripty run audio.mp3          # Transcribe audio
transcripty benchmark audio.mp3    # Benchmark models
```

## Belangrijke paden

- `transcripty/` — Package root
  - `config.py` — Pydantic Settings config (YAML + env vars + runtime overrides)
  - `hardware.py` — Hardware detection, optimal settings suggestion
  - `models.py` — Pydantic models (Word, Segment, TranscriptionResult, etc.)
  - `transcribe.py` — Whisper wrapper met thread-safe model caching
  - `diarize.py` — Pyannote wrapper met thread-safe pipeline caching
  - `merge.py` — Binary search merge (transcriptie + diarization)
  - `speakers.py` — Voice embedding DB met exclusive greedy matching
  - `pipeline.py` — `transcribe_with_speakers()` convenience functie
  - `formatters.py` — Output formatters (SRT, VTT, plain text)
  - `cli.py` — Click CLI (hardware, config, run, benchmark)
  - `device.py` — Device detection compat wrapper → hardware.py
  - `audio.py` — WAV conversie context manager
  - `vocabulary.py` — Domain vocabulary voor Whisper prompts
  - `py.typed` — PEP 561 type marker
- `tests/` — 97 pytest tests
- `benchmarks/` — Standalone benchmark scripts (niet in package)
- `pyproject.toml` — Package config, dependencies, CLI entry point

## Business rules

- Diarization is optioneel — base install werkt zonder torch/pyannote
- Config prioriteit: programmatic > env vars (TRANSCRIPTY\_\*) > YAML (~/.transcripty/config.yaml) > hardware defaults
- Thread safety: model/pipeline caches beschermd met threading.Lock
- Cache eviction: max_cached_models (default 2) — oldest-first eviction
- `diarize()` raises ValueError zonder HF token (niet alleen warning)
- Early validation: FileNotFoundError voor ontbrekende audio files
- Speaker matching is exclusive: 1 profiel per speaker, hoogste score eerst
- `UNKNOWN_SPEAKER` constante in models.py voor unmapped speakers

## Agent hints

- **data-engineer**: Geen database in dit project, pure library
- **reviewer**: Let op thread safety in caches, lazy import pattern, config singleton
- **qa**: Tests draaien zonder GPU/models via mocks, conftest.py reset singletons
- **pm**: Package is dependency voor Plaude (FastAPI), thread safety is kritiek
