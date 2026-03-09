"""Tests for CLI commands."""

from click.testing import CliRunner

from transcripty.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Transcripty" in result.output


def test_hardware_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["hardware"])
    assert result.exit_code == 0
    assert "CPU:" in result.output
    assert "Cores:" in result.output
    assert "RAM:" in result.output


def test_config_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["config"])
    assert result.exit_code == 0
    assert "model_size:" in result.output
    assert "compute_type:" in result.output


def test_run_missing_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "nonexistent.wav"])
    assert result.exit_code != 0


def test_benchmark_missing_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark", "nonexistent.wav"])
    assert result.exit_code != 0
