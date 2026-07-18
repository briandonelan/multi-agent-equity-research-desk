"""Smoke tests: the package imports and the CLI is wired up."""

from __future__ import annotations

from typer.testing import CliRunner

from desk.cli import app

runner = CliRunner()


def test_cli_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "research desk" in result.output.lower()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
