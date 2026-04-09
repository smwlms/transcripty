"""Gladia Benchmark Comparison — Run Gladia's exact datasets with transcripty.

Reproduces the evaluation from Gladia's Speech Recognition Benchmark Report 2026
using the same HuggingFace datasets, same normalization (gladia-normalization),
and same WER metric for a direct apples-to-apples comparison.

Requirements (benchmark-only, not part of transcripty core):
    pip install transcripty[benchmark]
    # or manually:
    pip install datasets jiwer gladia-normalization soundfile

Usage:
    PYTHONPATH=. python benchmarks/gladia_benchmark.py
    PYTHONPATH=. python benchmarks/gladia_benchmark.py --dataset voxpopuli_cleaned
    PYTHONPATH=. python benchmarks/gladia_benchmark.py --dataset mls --language de
    PYTHONPATH=. python benchmarks/gladia_benchmark.py --max-samples 50
    PYTHONPATH=. python benchmarks/gladia_benchmark.py --model large-v3 --report

Datasets (same as Gladia benchmark):
    voxpopuli_cleaned  — VoxPopuli Cleaned AA (628 samples, EN)
    earnings22_cleaned — Earnings22 Cleaned AA (6 samples, EN)
    earnings22_full    — Earnings22 Full (EN, long-form)
    mls                — Multilingual LibriSpeech (DE/ES/FR/IT/PT)
    switchboard        — Switchboard conversational speech (EN)
    pipecat            — Pipecat STT Benchmark (1000 samples, EN)

Reference:
    https://gladia.io/competitors/benchmarks
    https://github.com/gladiaio/normalization
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Gladia's published WER results (%) — Benchmark Report 2026
# ---------------------------------------------------------------------------
GLADIA_REFERENCE: dict[str, dict[str, float]] = {
    "voxpopuli_cleaned": {
        "gladia-solaria-1": 2.20,
        "assemblyai-universal-3-pro": 2.10,
        "assemblyai-universal-2": 2.20,
        "elevenlabs-scribe_v2": 1.70,
        "deepgram-nova-3": 3.20,
        "speechmatics": 3.00,
        "mistralai-voxtral": 2.10,
    },
    "earnings22_cleaned": {
        "gladia-solaria-1": 7.90,
        "assemblyai-universal-3-pro": 7.00,
        "assemblyai-universal-2": 6.90,
        "elevenlabs-scribe_v2": 7.90,
        "deepgram-nova-3": 12.70,
        "speechmatics": 7.70,
        "soniox-v4": 5.70,
        "mistralai-voxtral": 7.50,
    },
    "earnings22_full": {
        "gladia-solaria-1": 11.80,
        "assemblyai-universal-3-pro": 11.00,
        "assemblyai-universal-2": 11.10,
        "elevenlabs-scribe_v2": 9.40,
        "deepgram-nova-3": 14.50,
        "speechmatics": 10.00,
        "mistralai-voxtral": 11.60,
    },
    "mls": {
        "gladia-solaria-1": 5.80,
        "assemblyai-universal-3-pro": 4.70,
        "assemblyai-universal-2": 6.20,
        "elevenlabs-scribe_v2": 3.70,
        "deepgram-nova-3": 7.50,
        "soniox-v4": 5.60,
    },
    "mls_de": {
        "gladia-solaria-1": 5.00,
        "assemblyai-universal-3-pro": 3.50,
        "elevenlabs-scribe_v2": 3.10,
        "deepgram-nova-3": 6.90,
        "soniox-v4": 5.40,
        "assemblyai-universal-2": 3.40,
    },
    "mls_es": {
        "gladia-solaria-1": 4.00,
        "assemblyai-universal-3-pro": 3.20,
        "elevenlabs-scribe_v2": 3.20,
        "deepgram-nova-3": 4.60,
        "soniox-v4": 4.40,
        "assemblyai-universal-2": 4.00,
    },
    "mls_fr": {
        "gladia-solaria-1": 4.80,
        "assemblyai-universal-3-pro": 2.60,
        "elevenlabs-scribe_v2": 2.90,
        "deepgram-nova-3": 6.20,
        "soniox-v4": 5.00,
        "assemblyai-universal-2": 5.80,
    },
    "mls_it": {
        "gladia-solaria-1": 9.90,
        "assemblyai-universal-3-pro": 9.70,
        "elevenlabs-scribe_v2": 6.10,
        "deepgram-nova-3": 8.80,
        "soniox-v4": 8.80,
        "assemblyai-universal-2": 11.90,
    },
    "mls_pt": {
        "gladia-solaria-1": 5.30,
        "assemblyai-universal-3-pro": 4.40,
        "elevenlabs-scribe_v2": 3.00,
        "deepgram-nova-3": 11.30,
        "soniox-v4": 4.30,
        "assemblyai-universal-2": 5.90,
    },
    "switchboard": {
        "gladia-solaria-1": 35.80,
        "assemblyai-universal-3-pro": 56.00,
        "assemblyai-universal-2": 63.10,
        "elevenlabs-scribe_v2": 62.50,
        "deepgram-nova-3": 65.20,
        "speechmatics": 56.00,
        "soniox-v4": 62.90,
        "mistralai-voxtral": 50.10,
    },
    "pipecat": {
        "gladia-solaria-1": 2.70,
        "assemblyai-universal-3-pro": 2.00,
        "assemblyai-universal-2": 2.50,
        "elevenlabs-scribe_v2": 2.20,
        "deepgram-nova-3": 3.10,
        "speechmatics": 2.70,
        "soniox-v4": 2.90,
        "mistralai-voxtral": 2.60,
    },
}


@dataclass
class Sample:
    id: str
    audio_path: str
    reference: str
    language: str


@dataclass
class DatasetConfig:
    name: str
    display_name: str
    hf_id: str
    split: str
    text_column: str
    audio_column: str
    language: str
    hf_config: str | None = None
    needs_repo_download: bool = False
    default_max_samples: int | None = None


# ---------------------------------------------------------------------------
# Dataset configurations — same sources as Gladia benchmark
# ---------------------------------------------------------------------------
DATASETS: dict[str, DatasetConfig] = {
    "voxpopuli_cleaned": DatasetConfig(
        name="voxpopuli_cleaned",
        display_name="VoxPopuli Cleaned AA",
        hf_id="ArtificialAnalysis/VoxPopuli-Cleaned-AA",
        split="test",
        text_column="transcript",
        audio_column="url",
        language="en",
        needs_repo_download=True,
    ),
    "earnings22_cleaned": DatasetConfig(
        name="earnings22_cleaned",
        display_name="Earnings22 Cleaned AA",
        hf_id="ArtificialAnalysis/Earnings22-Cleaned-AA",
        split="test",
        text_column="transcript",
        audio_column="url",
        language="en",
        needs_repo_download=True,
    ),
    "earnings22_full": DatasetConfig(
        name="earnings22_full",
        display_name="Earnings22 Full",
        hf_id="revdotcom/earnings22",
        split="test",
        text_column="text",
        audio_column="audio",
        language="en",
    ),
    "switchboard": DatasetConfig(
        name="switchboard",
        display_name="Switchboard",
        hf_id="hhoangphuoc/switchboard",
        split="test",
        text_column="transcript",
        audio_column="audio",
        language="en",
        default_max_samples=200,
    ),
    "pipecat": DatasetConfig(
        name="pipecat",
        display_name="Pipecat STT",
        hf_id="pipecat-ai/stt-benchmark-data",
        split="train",
        text_column="transcription",
        audio_column="audio",
        language="en",
    ),
}

# MLS per-language configs
_MLS_LANGUAGES = {
    "german": "de",
    "spanish": "es",
    "french": "fr",
    "italian": "it",
    "portuguese": "pt",
}
for _lang_name, _code in _MLS_LANGUAGES.items():
    DATASETS[f"mls_{_code}"] = DatasetConfig(
        name=f"mls_{_code}",
        display_name=f"MLS {_lang_name.title()}",
        hf_id="facebook/multilingual_librispeech",
        hf_config=_lang_name,
        split="test",
        text_column="text",
        audio_column="audio",
        language=_code,
        default_max_samples=200,
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
_norm_pipelines: dict[str, object] = {}


def normalize_text(text: str, language: str, use_gladia: bool) -> str:
    """Normalize text for WER comparison.

    Uses gladia-normalization (gladia-3 preset) when available,
    matching Gladia's exact methodology. Falls back to basic
    lowercasing + punctuation removal otherwise.
    """
    if use_gladia:
        from normalization import load_pipeline

        key = language
        if key not in _norm_pipelines:
            _norm_pipelines[key] = load_pipeline("gladia-3", language=language)
        return _norm_pipelines[key].normalize(text)

    # Basic fallback — not directly comparable to Gladia's results
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def _load_audio_dataset(config: DatasetConfig, max_samples: int | None) -> list[Sample]:
    """Load dataset with embedded audio column from HuggingFace.

    Uses Parquet-level access to extract raw audio bytes, avoiding the
    need for torchcodec/torch at decode time.
    """
    from datasets import load_dataset
    from datasets.features import Audio

    effective_max = max_samples or config.default_max_samples

    print(f"  Downloading {config.display_name} from HuggingFace...")
    kwargs: dict = {}
    if config.hf_config:
        kwargs["name"] = config.hf_config

    ds = load_dataset(config.hf_id, split=config.split, **kwargs)

    # Disable audio decoding so we get raw file bytes/paths instead
    if config.audio_column in ds.features and isinstance(ds.features[config.audio_column], Audio):
        ds = ds.cast_column(config.audio_column, Audio(decode=False))

    if effective_max and len(ds) > effective_max:
        print(f"  Sampling {effective_max} of {len(ds)} rows...")
        ds = ds.select(range(effective_max))

    print(f"  Preparing {len(ds)} samples...")
    tmp_dir = tempfile.mkdtemp(prefix=f"gladia_{config.name}_")
    samples: list[Sample] = []

    for i, item in enumerate(ds):
        reference = item[config.text_column]
        if not reference or not reference.strip():
            continue

        audio = item[config.audio_column]
        audio_path = _resolve_audio(audio, tmp_dir, i)

        if audio_path:
            samples.append(
                Sample(
                    id=f"{config.name}_{i}",
                    audio_path=audio_path,
                    reference=reference,
                    language=config.language,
                )
            )

    return samples


def _resolve_audio(audio, tmp_dir: str, index: int) -> str | None:
    """Extract a usable file path from an HF audio item.

    Handles both decoded dicts (path/array) and non-decoded dicts
    (path + bytes).
    """
    import soundfile as sf

    if isinstance(audio, str) and Path(audio).exists():
        return audio

    if not isinstance(audio, dict):
        return None

    # Non-decoded: dict with "path" and "bytes"
    if audio.get("bytes"):
        ext = Path(audio.get("path") or "audio.wav").suffix or ".wav"
        tmp_path = os.path.join(tmp_dir, f"sample_{index}{ext}")
        with open(tmp_path, "wb") as f:
            f.write(audio["bytes"])
        return tmp_path

    # Path to cached file on disk
    if audio.get("path") and Path(str(audio["path"])).exists():
        return str(audio["path"])

    # Decoded: dict with "array" and "sampling_rate"
    if "array" in audio and audio["array"] is not None:
        tmp_path = os.path.join(tmp_dir, f"sample_{index}.wav")
        sf.write(tmp_path, audio["array"], audio["sampling_rate"])
        return tmp_path

    return None


def _load_repo_dataset(config: DatasetConfig, max_samples: int | None) -> list[Sample]:
    """Load dataset where audio files are stored in the HF repo."""
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    effective_max = max_samples or config.default_max_samples

    print(f"  Downloading {config.display_name} repo...")
    repo_path = snapshot_download(repo_id=config.hf_id, repo_type="dataset")

    print("  Loading metadata...")
    ds = load_dataset(config.hf_id, split=config.split)

    if effective_max and len(ds) > effective_max:
        print(f"  Sampling {effective_max} of {len(ds)} rows...")
        ds = ds.select(range(effective_max))

    samples: list[Sample] = []
    for i, item in enumerate(ds):
        reference = item[config.text_column]
        if not reference or not reference.strip():
            continue

        audio_rel = item[config.audio_column]
        audio_path = os.path.join(repo_path, audio_rel)

        if not os.path.exists(audio_path):
            print(f"    SKIP sample {i}: audio not found at {audio_rel}")
            continue

        samples.append(
            Sample(
                id=f"{config.name}_{i}",
                audio_path=audio_path,
                reference=reference,
                language=config.language,
            )
        )

    return samples


def load_samples(config: DatasetConfig, max_samples: int | None) -> list[Sample]:
    """Load samples from a dataset configuration."""
    if config.needs_repo_download:
        return _load_repo_dataset(config, max_samples)
    return _load_audio_dataset(config, max_samples)


# ---------------------------------------------------------------------------
# WER computation
# ---------------------------------------------------------------------------
def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute WER using jiwer."""
    import jiwer

    if not reference.strip():
        return {
            "wer": 0.0 if not hypothesis.strip() else 1.0,
            "wer_pct": 0.0 if not hypothesis.strip() else 100.0,
            "substitutions": 0,
            "insertions": len(hypothesis.split()) if hypothesis.strip() else 0,
            "deletions": 0,
            "ref_words": 0,
            "hyp_words": len(hypothesis.split()),
        }

    result = jiwer.process_words(reference, hypothesis)
    return {
        "wer": round(result.wer, 4),
        "wer_pct": round(result.wer * 100, 2),
        "substitutions": result.substitutions,
        "insertions": result.insertions,
        "deletions": result.deletions,
        "ref_words": len(reference.split()),
        "hyp_words": len(hypothesis.split()),
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(
    dataset_names: list[str],
    model_size: str = "large-v3-turbo",
    max_samples: int | None = None,
    language_override: str | None = None,
    generate_report: bool = False,
    output_path: str | None = None,
) -> dict:
    """Run the Gladia-comparable benchmark."""
    from transcripty import transcribe

    has_gladia_norm = _check_deps()

    # Expand meta-names
    resolved: list[str] = []
    for name in dataset_names:
        if name == "all":
            resolved = list(DATASETS.keys())
            break
        if name == "mls":
            resolved.extend(f"mls_{c}" for c in ["de", "es", "fr", "it", "pt"])
        else:
            resolved.append(name)

    all_results: dict[str, dict] = {}

    for ds_name in resolved:
        if ds_name not in DATASETS:
            print(f"Unknown dataset: {ds_name}")
            print(f"Available: {', '.join(sorted(DATASETS.keys()))}")
            continue

        config = DATASETS[ds_name]
        lang = language_override or config.language

        print(f"\n{'=' * 70}")
        print(f"  {config.display_name}")
        print(f"  HF: {config.hf_id} | split={config.split} | lang={lang}")
        print(f"  Model: {model_size}")
        print(f"{'=' * 70}")

        try:
            samples = load_samples(config, max_samples)
        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            continue

        if not samples:
            print("  No samples loaded, skipping.")
            continue

        print(f"  Loaded {len(samples)} samples\n")

        sample_results: list[dict] = []
        total_ref_words = 0
        total_errors = 0
        total_audio_s = 0.0
        total_wall_s = 0.0

        for i, sample in enumerate(samples):
            tag = f"[{i + 1}/{len(samples)}]"
            print(f"  {tag} {sample.id} ...", end="", flush=True)

            try:
                t0 = time.time()
                r = transcribe(
                    sample.audio_path,
                    model_size=model_size,
                    language=lang,
                    word_timestamps=False,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    temperature=0.0,
                )
                dt = time.time() - t0

                hypothesis = " ".join(seg.text for seg in r.segments).strip()

                ref_norm = normalize_text(sample.reference, lang, has_gladia_norm)
                hyp_norm = normalize_text(hypothesis, lang, has_gladia_norm)

                wer = compute_wer(ref_norm, hyp_norm)

                total_ref_words += wer["ref_words"]
                total_errors += wer["substitutions"] + wer["insertions"] + wer["deletions"]
                total_audio_s += r.duration
                total_wall_s += dt
                rtf = r.duration / dt if dt > 0 else 0

                sample_results.append(
                    {
                        "id": sample.id,
                        "wer": wer,
                        "rtf": round(rtf, 1),
                        "duration_s": round(r.duration, 1),
                        "wall_s": round(dt, 1),
                    }
                )
                print(f" WER={wer['wer_pct']:.1f}%  RTFx={rtf:.1f}")

            except Exception as e:
                print(f" ERROR: {e}")
                sample_results.append({"id": sample.id, "error": str(e)})

        # ---- Aggregate ----
        ok = [r for r in sample_results if "wer" in r]
        if not ok:
            print("  No successful transcriptions.")
            continue

        avg_wer = round(total_errors / total_ref_words * 100, 2) if total_ref_words else 0
        avg_rtf = round(total_audio_s / total_wall_s, 1) if total_wall_s else 0
        perfect = sum(1 for r in ok if r["wer"]["wer_pct"] == 0)
        high_wer = sum(1 for r in ok if r["wer"]["wer_pct"] > 100)

        ds_result = {
            "dataset": ds_name,
            "display_name": config.display_name,
            "model": model_size,
            "language": lang,
            "num_samples": len(ok),
            "avg_wer": avg_wer,
            "avg_rtf": avg_rtf,
            "perfect": perfect,
            "high_wer": high_wer,
            "total_audio_hours": round(total_audio_s / 3600, 2),
            "total_ref_words": total_ref_words,
            "total_errors": total_errors,
            "samples": sample_results,
        }
        all_results[ds_name] = ds_result

        # Print summary + comparison
        print(f"\n{'─' * 50}")
        print(f"  {config.display_name} — Summary")
        print(f"  Average WER : {avg_wer}%")
        print(f"  RTFx        : {avg_rtf}x")
        print(f"  Samples     : {len(ok)}  |  Perfect: {perfect}  |  High WER: {high_wer}")
        print(f"  Audio       : {total_audio_s / 3600:.2f} h")

        ref = GLADIA_REFERENCE.get(ds_name, {})
        if ref:
            print(f"\n  {'Provider':<35} {'WER':>8}")
            print(f"  {'─' * 45}")
            print(f"  {'transcripty (' + model_size + ')':<35} {avg_wer:>7.2f}%")
            for prov, wer_pct in sorted(ref.items(), key=lambda x: x[1]):
                print(f"  {prov:<35} {wer_pct:>7.2f}%")
        print(f"{'─' * 50}")

    # ---- Save results ----
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model_size,
        "normalization": "gladia-3" if has_gladia_norm else "basic-fallback",
        "datasets": {
            k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in all_results.items()
        },
        "detailed_samples": {k: v["samples"] for k, v in all_results.items()},
    }

    json_path = output_path or str(RESULTS_DIR / "gladia_benchmark_results.json")
    if not json_path.endswith(".json"):
        json_path = str(RESULTS_DIR / "gladia_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults JSON: {json_path}")

    if generate_report and all_results:
        report = _build_report(all_results, model_size, has_gladia_norm)
        report_path = str(RESULTS_DIR / "gladia_comparison_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report:       {report_path}")

    return all_results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def _build_report(results: dict, model_size: str, has_gladia_norm: bool) -> str:
    lines = [
        "# Transcripty vs Gladia Benchmark Comparison",
        "",
        f"**Model:** {model_size}  ",
        f"**Date:** {time.strftime('%Y-%m-%d')}  ",
        "**Normalization:** "
        + ("gladia-3 (identical to Gladia)" if has_gladia_norm else "basic fallback")
        + "  ",
        "",
        "## Overall WER Summary",
        "",
        "| Dataset | Transcripty | Best Competitor | Gladia solaria-1 | Samples |",
        "|---------|------------|-----------------|-------------------|---------|",
    ]

    for ds_name, res in results.items():
        ref = GLADIA_REFERENCE.get(ds_name, {})
        best = f"{min(ref.values()):.2f}%" if ref else "—"
        gladia = f"{ref['gladia-solaria-1']:.2f}%" if "gladia-solaria-1" in ref else "—"
        lines.append(
            f"| {res['display_name']} | **{res['avg_wer']:.2f}%** "
            f"| {best} | {gladia} | {res['num_samples']} |"
        )

    lines.extend(["", "## Per-Dataset Comparison", ""])

    for ds_name, res in results.items():
        ref = GLADIA_REFERENCE.get(ds_name, {})
        lines.extend(
            [
                f"### {res['display_name']}",
                "",
                f"- **Samples:** {res['num_samples']}",
                f"- **Audio:** {res['total_audio_hours']} hours",
                f"- **RTFx:** {res['avg_rtf']}x",
                f"- **Perfect:** {res['perfect']}  |  **High WER:** {res['high_wer']}",
                "",
                "| Provider | WER |",
                "|----------|-----|",
                f"| **transcripty ({model_size})** | **{res['avg_wer']:.2f}%** |",
            ]
        )
        for prov, wer_pct in sorted(ref.items(), key=lambda x: x[1]):
            lines.append(f"| {prov} | {wer_pct:.2f}% |")
        lines.append("")

    lines.extend(
        [
            "## Methodology",
            "",
            "Reproduces [Gladia's Speech Recognition Benchmark Report 2026]"
            "(https://gladia.io/competitors/benchmarks):",
            "",
            "- **Same audio** — identical HuggingFace datasets & test splits",
            "- **Same normalization** — `gladia-normalization` (`gladia-3` preset)",
            "- **Same metric** — Word Error Rate (WER)",
            "",
            "### Key differences",
            "",
            "- Gladia benchmarks cloud APIs; transcripty runs locally with faster-whisper",
            "- RTFx depends on local hardware, not API latency",
            "- Language is passed explicitly (matching Gladia's methodology)",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
def _check_deps() -> bool:
    """Return True if gladia-normalization is available."""
    missing = []
    for pkg in ("datasets", "jiwer", "soundfile"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    try:
        from normalization import load_pipeline  # noqa: F401

        return True
    except ImportError:
        print("WARNING: gladia-normalization not installed.")
        print("  Results won't be directly comparable to Gladia's benchmark.")
        print("  Install with: pip install gladia-normalization\n")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gladia Benchmark Comparison — transcripty vs cloud STT providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
datasets:
  all                 Run all datasets
  voxpopuli_cleaned   VoxPopuli Cleaned AA (628 samples, EN)
  earnings22_cleaned  Earnings22 Cleaned AA (6 samples, EN)
  earnings22_full     Earnings22 Full (EN, long-form)
  mls                 All MLS languages (DE/ES/FR/IT/PT)
  mls_de .. mls_pt    Individual MLS language
  switchboard         Switchboard conversational (EN)
  pipecat             Pipecat STT (1000 samples, EN)
""",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        nargs="+",
        default=["all"],
        help="Dataset(s) to benchmark (default: all)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="large-v3-turbo",
        help="Whisper model size (default: large-v3-turbo)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per dataset (overrides defaults)",
    )
    parser.add_argument("--language", default=None, help="Override language for all datasets")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--output", "-o", default=None, help="Output path for JSON results")

    args = parser.parse_args()
    run_benchmark(
        dataset_names=args.dataset,
        model_size=args.model,
        max_samples=args.max_samples,
        language_override=args.language,
        generate_report=args.report,
        output_path=args.output,
    )
