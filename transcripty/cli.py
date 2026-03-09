"""CLI entry point for Transcripty."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import click


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    """Transcripty — audio transcription & diarization toolkit."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(name)s %(levelname)s: %(message)s")


@cli.command()
def hardware():
    """Show detected hardware and recommended settings."""
    from transcripty.hardware import detect_hardware

    profile = detect_hardware()
    suggestions = profile.suggest_settings()

    click.echo(f"CPU:      {profile.cpu}")
    click.echo(f"Cores:    {profile.cores}")
    click.echo(f"RAM:      {profile.ram_gb:.1f} GB")
    click.echo(f"GPU:      {profile.gpu or 'None'}")
    click.echo(f"MPS:      {profile.mps}")
    click.echo(f"Device:   {profile.device}")
    click.echo()
    click.echo("Recommended settings:")
    for key, value in suggestions.items():
        click.echo(f"  {key}: {value}")


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--models", default="small,medium", help="Comma-separated model sizes")
@click.option("--language", default=None, help="Language code (e.g. 'nl', 'en')")
@click.option("--output", "-o", default=None, help="JSON output path")
def benchmark(audio_path: str, models: str, language: str | None, output: str | None):
    """Benchmark transcription speed across models."""
    from transcripty.transcribe import transcribe

    model_list = [m.strip() for m in models.split(",")]
    results = []

    for model_size in model_list:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Model: {model_size}")
        click.echo(f"{'=' * 60}")

        start = time.time()
        result = transcribe(
            audio_path,
            model_size=model_size,  # type: ignore[arg-type]
            language=language,
        )
        elapsed = round(time.time() - start, 2)
        rt_factor = round(elapsed / result.duration, 2) if result.duration > 0 else 0

        entry = {
            "model": model_size,
            "time_s": elapsed,
            "rt_factor": rt_factor,
            "segments": len(result.segments),
            "language": result.language,
            "duration_s": round(result.duration, 1),
            "preview": result.segments[0].text[:200] if result.segments else "",
        }
        results.append(entry)

        click.echo(f"Time:     {elapsed}s")
        click.echo(f"RT:       {rt_factor}x")
        click.echo(f"Segments: {len(result.segments)}")
        click.echo(f"Language: {result.language}")
        if result.segments:
            click.echo(f"Preview:  {result.segments[0].text[:100]}...")

    # Summary table
    if len(results) > 1:
        click.echo(f"\n{'=' * 60}")
        click.echo("COMPARISON")
        click.echo(f"{'Model':<18} {'Time':>8} {'RT':>8} {'Segments':>10}")
        click.echo("-" * 50)
        for r in results:
            click.echo(
                f"{r['model']:<18} {r['time_s']:>7}s {r['rt_factor']:>7}x " f"{r['segments']:>10}"
            )

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to {output}")


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--model", default=None, help="Override model size")
@click.option("--language", default=None, help="Language code")
@click.option(
    "--compute-type",
    default=None,
    type=click.Choice(["int8", "float16", "float32", "auto"]),
    help="Compute type (default: from config)",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "srt", "vtt"]),
    default="text",
    help="Output format",
)
@click.option("--diarize/--no-diarize", default=False, help="Enable speaker diarization")
@click.option("--output", "-o", default=None, help="Output file path (default: stdout)")
def run(
    audio_path: str,
    model: str | None,
    language: str | None,
    compute_type: str | None,
    fmt: str,
    diarize: bool,
    output: str | None,
):
    """Transcribe an audio file."""
    from transcripty.formatters import to_srt, to_text, to_vtt

    kwargs: dict = {}
    if model:
        kwargs["model_size"] = model
    if language:
        kwargs["language"] = language
    if compute_type:
        kwargs["compute_type"] = compute_type

    if diarize:
        from transcripty.pipeline import transcribe_with_speakers

        segments = transcribe_with_speakers(audio_path, **kwargs)
    else:
        from transcripty.transcribe import transcribe

        result = transcribe(audio_path, **kwargs)
        segments = result.segments

    # Format output
    if fmt == "srt":
        text = to_srt(segments)
    elif fmt == "vtt":
        text = to_vtt(segments)
    else:
        text = to_text(segments, include_speakers=diarize)

    if output:
        Path(output).write_text(text, encoding="utf-8")
        click.echo(f"Output written to {output}")
    else:
        click.echo(text)


@cli.command()
def config():
    """Show current configuration."""
    from transcripty.config import get_config

    cfg = get_config()
    for key, value in cfg.model_dump().items():
        click.echo(f"{key}: {value}")


if __name__ == "__main__":
    cli()
