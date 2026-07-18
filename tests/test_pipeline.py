"""Multi-agent DAG (offline): full flow, critic isolation, and failure propagation."""

from __future__ import annotations

from desk.agents.base import AgentResult, CallbackAgentRunner
from desk.data import sections
from desk.orchestrator.multi_agent import run_pipeline
from desk.orchestrator.run import RunContext

# Three in-universe industrials with one seeded filing section each.
TICKERS = {
    "CAT": ("0000000021-25-000001", "Caterpillar Inc."),
    "DE": ("0000000022-25-000001", "Deere & Company"),
    "HON": ("0000000023-25-000001", "Honeywell International Inc."),
}


def _seed_sections():
    for _, (acc, _name) in TICKERS.items():
        text = (
            "Revenue increased 4 percent year over year driven by higher volumes."
            "\n\n"
            "Operating margin improved while management flagged input cost pressure as a risk."
        )
        sections.store_section_text(acc, "7", text)


def _screen_json():
    return {
        "interpreted_intent": "profitable industrials with improving margins",
        "filters_applied": [{"field": "operating_margin", "op": ">", "value": 0.1}],
        "candidates": [
            {
                "ticker": t,
                "cik": "1",
                "company_name": name,
                "rationale": "Solid margins.",
                "metrics_snapshot": {"operating_margin": 0.16},
            }
            for t, (_, name) in TICKERS.items()
        ],
    }


def _thesis_json(ticker, acc):
    cite = {"source_ref": f"{acc}#7¶0", "quote": "Revenue increased 4 percent year over year"}
    claim = {"text": "Revenue grew.", "claim_type": "fact", "citations": [cite]}
    return {
        "ticker": ticker,
        "thesis_summary": "Bullish on volumes.",
        "claims": [claim, claim, claim, claim],
    }


def _critique_json(ticker, acc):
    cite = {"source_ref": f"{acc}#7¶1", "quote": "management flagged input cost pressure as a risk"}
    ch = {
        "target_claim_idx": 0,
        "text": "Costs pressure margins.",
        "severity": "material",
        "citations": [cite],
    }
    return {
        "ticker": ticker,
        "challenges": [ch, ch],
        "overall_assessment": "thesis_weakened",
        "what_would_change_my_mind": "Sustained margin expansion over 3 quarters.",
    }


def _memo_json(ticker, name, acc, bear=True):
    bull_cite = {"source_ref": f"{acc}#7¶0", "quote": "Revenue increased 4 percent year over year"}
    bear_cite = {
        "source_ref": f"{acc}#7¶1",
        "quote": "management flagged input cost pressure as a risk",
    }
    return {
        "ticker": ticker,
        "company_name": name,
        "thesis_summary": "Balanced view.",
        "valuation_snapshot": {"operating_margin": 0.16},
        "bull_case": [{"text": "Revenue grew.", "claim_type": "fact", "citations": [bull_cite]}],
        "bear_case": (
            [
                {
                    "target_claim_idx": 0,
                    "text": "Costs pressure margins.",
                    "severity": "material",
                    "citations": [bear_cite],
                }
            ]
            if bear
            else []
        ),
        "unresolved_disagreements": ["Whether margin gains persist."],
        "confidence": "medium",
        "confidence_rationale": "One material challenge.",
    }


def _ticker_in(prompt: str) -> str:
    for t in TICKERS:
        if t in prompt:
            return t
    return ""


def _responder(spec):
    t = _ticker_in(spec.prompt)
    acc = TICKERS.get(t, ("ACC", ""))[0]
    name = TICKERS.get(t, ("", "X"))[1]
    if spec.stage == "screener":
        return _screen_json()
    if spec.stage == "reader":
        return _thesis_json(t, acc)
    if spec.stage == "critic":
        return _critique_json(t, acc)
    if spec.stage == "synthesizer":
        has_critique = "UNAVAILABLE" not in spec.prompt
        return _memo_json(t, name, acc, bear=has_critique)
    return {}


async def test_full_pipeline_three_memos(mock_edgar):
    _seed_sections()
    runner = CallbackAgentRunner(_responder)
    ctx = RunContext("PIPE1", engine="pipeline", query="q")
    memos, failures = await run_pipeline(
        "profitable industrials", run_id="PIPE1", runner=runner, run_ctx=ctx, max_candidates=3
    )
    assert failures == []
    assert len(memos) == 3
    assert all(m.bull_case and m.bear_case for m in memos)
    # Every handoff artifact was persisted.
    names = {p.name for p in (ctx.root / "handoffs").glob("*.json")}
    assert "screen_result.json" in names
    assert any(n.startswith("thesis_") for n in names)
    assert any(n.startswith("critique_") for n in names)


async def test_critic_isolation_default_contract(mock_edgar):
    _seed_sections()
    runner = CallbackAgentRunner(_responder)
    ctx = RunContext("PIPE2", engine="pipeline", query="q")
    await run_pipeline("x", run_id="PIPE2", runner=runner, run_ctx=ctx, max_candidates=1)
    critic_specs = [c for c in runner.calls if c.stage == "critic"]
    assert critic_specs
    # Under contract handoff mode, the critic must NOT see the reader's thesis_summary.
    assert "Bullish on volumes" not in critic_specs[0].prompt
    assert "analyst_thesis_summary" not in critic_specs[0].prompt


async def test_critic_isolation_off_in_full_context(mock_edgar):
    _seed_sections()
    runner = CallbackAgentRunner(_responder)
    ctx = RunContext("PIPE3", engine="pipeline", query="q")
    await run_pipeline(
        "x",
        run_id="PIPE3",
        runner=runner,
        run_ctx=ctx,
        max_candidates=1,
        handoff_mode="full_context",
    )
    critic_specs = [c for c in runner.calls if c.stage == "critic"]
    assert "analyst_thesis_summary" in critic_specs[0].prompt
    assert "Bullish on volumes" in critic_specs[0].prompt


async def test_reader_failure_drops_ticker(mock_edgar):
    _seed_sections()

    def responder(spec):
        if spec.stage == "reader" and "CAT" in spec.prompt:
            return AgentResult(text='{"ticker":"CAT"}')  # invalid -> HandoffFailure after repair
        return _responder(spec)

    runner = CallbackAgentRunner(responder)
    ctx = RunContext("PIPE4", engine="pipeline", query="q")
    memos, failures = await run_pipeline(
        "x", run_id="PIPE4", runner=runner, run_ctx=ctx, max_candidates=3
    )
    tickers = {m.ticker for m in memos}
    assert "CAT" not in tickers  # dropped
    assert len(memos) == 2  # DE, HON survive
    assert any("reader:CAT" in f.get("stage", "") for f in failures)


async def test_critic_failure_yields_memo_without_bear_case(mock_edgar):
    _seed_sections()

    def responder(spec):
        if spec.stage == "critic":
            return AgentResult(text="garbage not json")  # critic fails
        return _responder(spec)

    runner = CallbackAgentRunner(responder)
    ctx = RunContext("PIPE5", engine="pipeline", query="q")
    memos, failures = await run_pipeline(
        "x", run_id="PIPE5", runner=runner, run_ctx=ctx, max_candidates=1
    )
    # Memo still produced; the synthesizer is told the critique is unavailable, and the
    # orchestrator appends an explicit note.
    assert len(memos) == 1
    assert any("critic" in f.get("stage", "") for f in failures)
    assert any("critique was unavailable" in d.lower() for d in memos[0].unresolved_disagreements)


async def test_screener_failure_produces_no_memos(mock_edgar):
    def responder(spec):
        return AgentResult(text="not json")  # screener fails immediately

    runner = CallbackAgentRunner(responder)
    ctx = RunContext("PIPE6", engine="pipeline", query="q")
    memos, failures = await run_pipeline(
        "x", run_id="PIPE6", runner=runner, run_ctx=ctx, max_candidates=3
    )
    assert memos == []
    assert failures and failures[0]["stage"] == "screener"
    assert (ctx.root / "failures" / "screener.json").exists()


async def test_run_memo_writes_result_marker(mock_edgar):
    """result.json is written last and records completion + failure count, so a killed run (no
    result.json) is distinguishable from a genuine empty screen (result.json, n_memos=0)."""
    import json

    from desk.orchestrator.pipeline import run_memo
    from desk.settings import get_settings

    _seed_sections()
    runner = CallbackAgentRunner(_responder)
    result = await run_memo("profitable industrials", engine="pipeline", runner=runner,
                            max_candidates=3)
    marker = get_settings().runs_dir / result.run_id / "result.json"
    assert marker.exists()
    payload = json.loads(marker.read_text("utf-8"))
    assert payload["status"] == "complete"
    assert payload["n_memos"] == len(result.memos)
    assert payload["n_failures"] == len(result.failures)
    assert sorted(payload["memo_tickers"]) == sorted(m.ticker for m in result.memos)


async def test_run_memo_empty_screen_still_marks_complete(mock_edgar):
    """A screen that yields zero memos is a real result, not a crash: it still writes result.json."""
    import json

    from desk.orchestrator.pipeline import run_memo
    from desk.settings import get_settings

    _seed_sections()

    def empty_responder(spec):
        if spec.stage == "screener":
            # No candidates -> min_length=1 fails validation twice -> HandoffFailure -> empty screen.
            return {"interpreted_intent": "i", "filters_applied": [], "candidates": []}
        return _responder(spec)

    runner = CallbackAgentRunner(empty_responder)
    result = await run_memo("nothing matches", engine="pipeline", runner=runner, max_candidates=3)
    marker = get_settings().runs_dir / result.run_id / "result.json"
    assert marker.exists()  # present despite zero memos -> a completed empty screen
    payload = json.loads(marker.read_text("utf-8"))
    assert payload["n_memos"] == 0 and payload["status"] == "complete"


async def test_baseline_error_recorded_as_empty_not_dropped(mock_edgar):
    """A baseline that errors (e.g. max turns) is a recordable empty-handed run, not a crash."""
    import json

    from desk.orchestrator.pipeline import run_memo
    from desk.settings import get_settings

    _seed_sections()

    def boom(spec):
        raise RuntimeError("Reached maximum number of turns (24)")

    result = await run_memo("some query", engine="baseline",
                            runner=CallbackAgentRunner(boom), max_candidates=2)
    assert result.memos == [] and len(result.failures) == 1
    marker = get_settings().runs_dir / result.run_id / "result.json"
    payload = json.loads(marker.read_text("utf-8"))
    assert payload["status"] == "complete" and payload["n_memos"] == 0 and payload["n_failures"] == 1


def test_timeouts_flagged_distinctly_from_contract_failures(tmp_path, monkeypatch):
    """A wall-clock timeout must be recorded as timeout=True, so it never masquerades as a
    genuine contract failure in the scoreboard; result.json must count n_timeouts."""
    import json

    from desk.contracts.v1 import HandoffFailure
    from desk.orchestrator.multi_agent import _fail_record
    from desk.orchestrator.run import RunContext
    from desk.settings import get_settings

    # A timeout is flagged; a contract failure is not.
    assert _fail_record("critic", TimeoutError())["timeout"] is True
    assert _fail_record("reader", TimeoutError())["timeout"] is True
    assert _fail_record("critic", HandoffFailure("critic", ["bad quote"], "raw"))["timeout"] is False

    ctx = RunContext("TORUN", engine="pipeline", query="q")
    ctx.write_result(
        n_memos=1,
        n_failures=2,
        memo_tickers=["CAT"],
        failures=[
            {"stage": "critic", "errors": ["TimeoutError: "], "timeout": True},
            {"stage": "reader", "errors": ["bad quote"], "timeout": False},
        ],
    )
    payload = json.loads((get_settings().runs_dir / "TORUN" / "result.json").read_text("utf-8"))
    assert payload["n_timeouts"] == 1
    assert payload["n_failures"] == 2
    assert [f["timeout"] for f in payload["failures"]] == [True, False]
