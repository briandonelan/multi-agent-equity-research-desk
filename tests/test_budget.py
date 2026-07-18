"""Budget controller + degradation ladder: each rung visibly triggers."""

from __future__ import annotations

import json

from desk.agents.base import AgentResult, CallbackAgentRunner
from desk.ledger import db
from desk.orchestrator.multi_agent import run_pipeline
from desk.orchestrator.policies import (
    RUNG_DROP_TIER,
    RUNG_LIMIT_FILINGS,
    RUNG_REDUCE_TRUNCATION,
    BudgetController,
    BudgetDecision,
    next_cheaper_tier,
    plan_stage,
)
from desk.orchestrator.run import RunContext
from tests.test_pipeline import _responder, _seed_sections

# --- Unit: controller + plan ----------------------------------------------------------------


def test_ladder_engages_cumulatively_over_soft_cap():
    caps = {"a": {"soft_cap": 100, "hard_cap": 10_000}}
    ctrl = BudgetController(
        caps, ladder=[RUNG_REDUCE_TRUNCATION, RUNG_DROP_TIER, RUNG_LIMIT_FILINGS]
    )
    # Under soft cap -> nothing.
    assert ctrl.evaluate("a", 50).degradations == []
    # Each subsequent over-soft evaluation engages one more rung.
    assert ctrl.evaluate("a", 200).degradations == [RUNG_REDUCE_TRUNCATION]
    assert ctrl.evaluate("a", 300).degradations == [RUNG_REDUCE_TRUNCATION, RUNG_DROP_TIER]
    assert ctrl.evaluate("a", 400).degradations == [
        RUNG_REDUCE_TRUNCATION,
        RUNG_DROP_TIER,
        RUNG_LIMIT_FILINGS,
    ]
    # Ladder is capped at its length.
    assert len(ctrl.evaluate("a", 999).degradations) == 3


def test_hard_cap_sets_budget_exhausted():
    caps = {"a": {"soft_cap": 100, "hard_cap": 500}}
    ctrl = BudgetController(caps)
    assert ctrl.evaluate("a", 400).budget_exhausted is False
    assert ctrl.evaluate("a", 600).budget_exhausted is True


def test_next_cheaper_tier():
    order = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8"]
    assert next_cheaper_tier("claude-opus-4-8", order) == "claude-sonnet-5"
    assert next_cheaper_tier("claude-sonnet-5", order) == "claude-haiku-4-5"
    assert next_cheaper_tier("claude-haiku-4-5", order) == "claude-haiku-4-5"  # clamp
    assert next_cheaper_tier("unknown", order) == "unknown"


def test_plan_stage_applies_each_rung():
    order = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8"]
    dec = BudgetDecision(degradations=[RUNG_REDUCE_TRUNCATION, RUNG_DROP_TIER, RUNG_LIMIT_FILINGS])
    plan = plan_stage(
        decision=dec,
        model="claude-opus-4-8",
        base_max_section_chars=12_000,
        tier_order=order,
        truncation_factor=0.5,
    )
    assert plan.max_section_chars == 6_000  # reduced truncation
    assert plan.model == "claude-sonnet-5"  # dropped one tier
    assert plan.max_filings == 2  # limited filings


# --- Integration: low soft cap visibly triggers each rung via FakeAgentRunner ----------------


def _usage_responder(spec):
    """Wrap the pipeline responder, attaching non-trivial usage so spend crosses the soft cap."""
    out = _responder(spec)
    text = out if isinstance(out, str) else json.dumps(out)
    return AgentResult(text=text, usage={"input_tokens": 200, "output_tokens": 5_000})


async def test_low_soft_cap_triggers_full_ladder_in_pipeline(mock_edgar):
    _seed_sections()
    # soft cap of 1 token means every stage after the first is "over budget".
    caps = {
        s: {"soft_cap": 1, "hard_cap": 10_000_000}
        for s in ("screener", "reader", "critic", "synthesizer")
    }
    controller = BudgetController(caps)
    runner = CallbackAgentRunner(_usage_responder)
    ctx = RunContext("BUD1", engine="pipeline", query="q")

    memos, failures = await run_pipeline(
        "x", run_id="BUD1", runner=runner, run_ctx=ctx, max_candidates=1, controller=controller
    )
    assert memos, f"expected a memo; failures={failures}"

    # Inspect the ledger: the ladder must have engaged each rung across the stages.
    with db.connect() as conn:
        rows = {
            r["stage"]: json.loads(r["degradations"] or "[]")
            for r in conn.execute(
                "SELECT stage, degradations FROM llm_calls WHERE run_id=?", ("BUD1",)
            )
        }
    all_degradations = {d for degs in rows.values() for d in degs}
    assert RUNG_REDUCE_TRUNCATION in all_degradations
    assert RUNG_DROP_TIER in all_degradations
    assert RUNG_LIMIT_FILINGS in all_degradations


async def test_tier_drop_changes_recorded_model(mock_edgar):
    _seed_sections()
    caps = {
        s: {"soft_cap": 1, "hard_cap": 10_000_000}
        for s in ("screener", "reader", "critic", "synthesizer")
    }
    runner = CallbackAgentRunner(_usage_responder)
    ctx = RunContext("BUD2", engine="pipeline", query="q")
    await run_pipeline(
        "x",
        run_id="BUD2",
        runner=runner,
        run_ctx=ctx,
        max_candidates=1,
        controller=BudgetController(caps),
    )
    # The critic (base Opus) should have been dropped a tier once drop_model_tier engaged.
    critic_calls = [c for c in runner.calls if c.stage == "critic"]
    assert critic_calls
    assert critic_calls[0].model != "claude-opus-4-8"


def test_all_runs_trend_table_data():
    db.record_llm_call(
        run_id="T1", stage="reader", model="claude-sonnet-5", input_tokens=100, output_tokens=50
    )
    db.record_llm_call(
        run_id="T2", stage="critic", model="claude-opus-4-8", input_tokens=200, output_tokens=80
    )
    from desk.ledger import report

    rolls = report.all_runs()
    ids = {r.run_id for r in rolls}
    assert {"T1", "T2"} <= ids
    assert all(r.total_cost_usd >= 0 for r in rolls)
