"""Ledger: pricing math, call recording, and roll-ups."""

from __future__ import annotations

from desk.ledger import db, report


def test_compute_cost_from_pricing():
    # Sonnet 5: $3/MTok in, $15/MTok out -> 1M in + 1M out = $18.00
    cost = db.compute_cost("claude-sonnet-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 18.0
    # Cache read is ~0.1x input.
    assert db.compute_cost("claude-sonnet-5", cache_read_tokens=1_000_000) == 0.3
    assert db.compute_cost("unknown-model", input_tokens=1_000_000) == 0.0


def test_record_and_rollup():
    db.record_llm_call(
        run_id="R1",
        stage="reader",
        model="claude-sonnet-5",
        input_tokens=10_000,
        output_tokens=2_000,
        cache_read_tokens=5_000,
    )
    db.record_llm_call(
        run_id="R1",
        stage="critic",
        model="claude-opus-4-8",
        input_tokens=8_000,
        output_tokens=1_500,
    )
    db.record_tool_call(
        run_id="R1",
        stage="reader",
        tool="mcp__filings__get_section",
        args_chars=50,
        result_chars=8000,
        truncated=True,
    )

    roll = report.run_rollup("R1")
    assert roll.n_llm_calls == 2
    assert roll.n_tool_calls == 1
    assert roll.n_truncated_tool_calls == 1
    assert {s.stage for s in roll.stages} == {"reader", "critic"}
    assert roll.total_cost_usd > 0
    # Cache savings should be positive because reader had cache reads.
    assert roll.cache_savings_usd > 0


def test_build_cost_summary_by_stage():
    db.record_llm_call(
        run_id="R2", stage="baseline", model="claude-sonnet-5", input_tokens=1000, output_tokens=500
    )
    summ = report.build_cost_summary("R2")
    assert summ.run_id == "R2"
    assert summ.n_calls == 1
    assert "baseline" in summ.by_stage_cost_usd
