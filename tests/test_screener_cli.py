"""M7.4: the `desk eval screener` command — model-alias parsing and offline re-render."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from desk.agents.base import CallbackAgentRunner
from desk.cli import app
from desk.evals.screener import harness, report
from desk.evals.screener.harness import resolve_models
from desk.evals.screener.suite import load_suite

runner = CliRunner()


def test_resolve_models_aliases_and_passthrough():
    assert resolve_models("default") == [None]
    assert resolve_models("") == [None]
    assert resolve_models("haiku,sonnet") == ["claude-haiku-4-5", "claude-sonnet-5"]
    assert resolve_models("claude-opus-4-8") == ["claude-opus-4-8"]  # full id passes through


def test_eval_screener_help_exits_zero():
    result = runner.invoke(app, ["eval", "screener", "--help"])
    assert result.exit_code == 0
    assert "--repeats" in result.output and "--report" in result.output


def test_eval_screener_report_missing_run_errors():
    result = runner.invoke(app, ["eval", "screener", "--report", "NOPE"])
    assert result.exit_code == 1
    assert "No saved eval" in result.output


async def _build_saved_run(runs_dir, eval_run_id="RERENDER1"):
    """Produce a real results.json on disk using the fake agent, for the re-render test."""
    q = next(x for x in load_suite() if x.id == "q01_smid_tech")

    def responder(spec):
        return {
            "interpreted_intent": "small and mid-cap technology",
            "filters_applied": [
                {"field": "sector", "op": "==", "value": "Technology"},
                {"field": "market_cap", "op": "<", "value": 10000000000},
            ],
            "candidates": [
                {"ticker": t, "cik": "1", "company_name": t, "rationale": "x"}
                for t in ("TEC3", "TEC6")
            ],
        }

    evals = await harness.run_matrix(
        [None], [q], runner=CallbackAgentRunner(responder), run_id=eval_run_id, repeats=1
    )
    eval_dir = runs_dir / "evals" / "screener" / eval_run_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {"eval_run_id": eval_run_id, "repeats": 1, "results": report.to_json(evals)}
    (eval_dir / "results.json").write_text(json.dumps(payload), "utf-8")
    return eval_dir


def test_eval_screener_report_rerenders_from_disk(temp_cache):
    import asyncio

    from desk.settings import get_settings

    runs_dir = get_settings().runs_dir
    eval_dir = asyncio.run(_build_saved_run(runs_dir))

    result = runner.invoke(app, ["eval", "screener", "--report", "RERENDER1"])
    assert result.exit_code == 0
    assert "Screener evaluation" in result.output
    assert "q01_smid_tech" in result.output
    # Re-render writes report.md next to the saved results.
    assert (eval_dir / "report.md").exists()
    assert "<!-- ANALYSIS -->" in (eval_dir / "report.md").read_text("utf-8")
