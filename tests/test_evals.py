"""Eval harness (offline): judge, planted errors + isolation delta, injection, compare table."""

from __future__ import annotations

import json

from desk.agents.base import AgentResult, CallbackAgentRunner
from desk.contracts.v1 import Citation, Claim, ResearchMemo, ThesisDraft
from desk.data import sections
from desk.evals import compare, failure_injection, judge, planted_errors

# --- shared fixtures ------------------------------------------------------------------------

ACC = "0000000021-25-000001"


def _seed():
    text = (
        "Revenue increased 4 percent in 2025 driven by higher volumes."
        "\n\n"
        "Operating margin improved while management flagged input cost pressure as a risk."
    )
    sections.store_section_text(ACC, "7", text)


def _draft() -> ThesisDraft:
    cite = Citation(source_ref=f"{ACC}#7¶0", quote="Revenue increased 4 percent in 2025")
    claim = Claim(text="Revenue increased 4 percent in 2025.", claim_type="fact", citations=[cite])
    return ThesisDraft(
        run_id="r", ticker="CAT", thesis_summary="Bullish.", claims=[claim, claim, claim, claim]
    )


def _memo() -> ResearchMemo:
    bull_cite = Citation(source_ref=f"{ACC}#7¶0", quote="Revenue increased 4 percent in 2025")
    bear_cite = Citation(
        source_ref=f"{ACC}#7¶1", quote="management flagged input cost pressure as a risk"
    )
    return ResearchMemo(
        run_id="r",
        ticker="CAT",
        company_name="Caterpillar Inc.",
        as_of="2026-07-15",
        thesis_summary="t",
        valuation_snapshot={},
        bull_case=[Claim(text="Rev grew.", claim_type="fact", citations=[bull_cite])],
        bear_case=[
            {
                "target_claim_idx": 0,
                "text": "Costs.",
                "severity": "material",
                "citations": [bear_cite],
            }
        ],
        unresolved_disagreements=["x"],
        confidence="medium",
        confidence_rationale="y",
        cost={"run_id": "r"},
    )


# --- judge ----------------------------------------------------------------------------------


def test_programmatic_citation_accuracy():
    _seed()
    acc = judge.programmatic_citation_accuracy(_memo())
    assert acc.n_citations == 2
    assert acc.n_verified == 2  # both quotes resolve + match
    assert acc.accuracy == 1.0


def test_judge_strips_provenance():
    md = judge._memo_for_judge(_memo())
    assert "run_id" not in md
    assert "cost" not in md.lower() or "This memo cost" not in md


async def test_judge_memo_scores(mock_edgar):
    _seed()
    runner = CallbackAgentRunner(
        lambda spec: {
            "grounding": 4,
            "argument_balance": 5,
            "specificity": 4,
            "readability": 5,
            "justification": "well grounded",
        }
    )
    score = await judge.judge_memo(_memo(), run_id="J1", runner=runner)
    assert score.citation_accuracy == 1.0
    assert score.grounding == 4 and score.readability == 5


# --- planted errors -------------------------------------------------------------------------


def test_corruptions_change_the_claim():
    for kind in planted_errors.CORRUPTIONS:
        corrupted, corr = planted_errors.corrupt_draft(_draft(), kind)
        assert corr.kind == kind
        original = _draft().claims[0]
        changed = corrupted.claims[0]
        # Either the text or a citation changed.
        assert (changed.text != original.text) or (
            changed.citations[0].source_ref != original.citations[0].source_ref
        )


async def test_critic_eval_catch_rate_and_isolation_delta(mock_edgar):
    _seed()

    # Fake critic that "catches" corrupted claim 0 when isolation is ON (contract mode: prompt has
    # no analyst_thesis_summary), but is swayed (misses) when isolation is OFF (full_context).
    def responder(spec):
        isolated = "analyst_thesis_summary" not in spec.prompt
        challenge = {
            "target_claim_idx": 0 if isolated else None,
            "text": "citation does not support the claim" if isolated else "looks fine overall",
            "severity": "material" if isolated else "minor",
            "citations": [
                {
                    "source_ref": f"{ACC}#7¶1",
                    "quote": "management flagged input cost pressure as a risk",
                }
            ],
        }
        return {
            "ticker": "CAT",
            "challenges": [challenge, challenge],
            "overall_assessment": "thesis_weakened",
            "what_would_change_my_mind": "more evidence",
        }

    runner = CallbackAgentRunner(responder)
    delta = await planted_errors.evaluate_isolation_delta([_draft()], run_id="C1", runner=runner)
    assert delta.isolation_on.overall_catch_rate == 1.0  # catches all with isolation ON
    assert delta.isolation_off.overall_catch_rate == 0.0  # misses all with isolation OFF
    assert delta.catch_rate_delta == 1.0
    # by-type breakdown present for each corruption.
    assert set(delta.isolation_on.by_type_catch) == set(planted_errors.CORRUPTIONS)
    # Timeout accounting: no timeouts with a fast fake; every call counted as completed.
    assert delta.isolation_on.n_timeouts == 0
    assert all(delta.isolation_on.by_type_completed[k] == 1 for k in planted_errors.CORRUPTIONS)
    assert all(delta.isolation_on.by_type_timeout[k] == 0 for k in planted_errors.CORRUPTIONS)


# --- failure injection ----------------------------------------------------------------------


def test_injection_faults_are_caught_at_the_boundary():
    r1 = failure_injection.run_injection("truncate_citations")
    assert r1.caught_by == "pydantic"  # Claim requires >=1 citation

    r2 = failure_injection.run_injection("schema_drift")
    assert r2.caught_by == "pydantic"  # missing required field

    r3 = failure_injection.run_injection("ambiguous_ticker")
    assert r3.caught_by == "semantic"  # out-of-universe ticker

    r4 = failure_injection.run_injection("tool_truncation")
    assert r4.caught_by == "tool"  # explicit truncation marker


def test_injection_tags_ledger():
    from desk.ledger import db

    failure_injection.run_injection("schema_drift", run_id="INJTAG")
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT injected_fault FROM tool_calls WHERE run_id=?", ("INJTAG",)
        ).fetchall()
    assert any(r["injected_fault"] == "schema_drift" for r in rows)


# --- compare --------------------------------------------------------------------------------


async def test_compare_table_both_engines(mock_edgar):
    _seed()

    # Build valid artifacts for both engines from one responder.
    def responder(spec):
        if spec.stage == "judge":
            return AgentResult(
                text=json.dumps(
                    {
                        "grounding": 4,
                        "argument_balance": 4,
                        "specificity": 4,
                        "readability": 4,
                        "justification": "ok",
                    }
                )
            )
        if spec.stage == "screener":
            return AgentResult(
                text=json.dumps(
                    {
                        "interpreted_intent": "i",
                        "filters_applied": [],
                        "candidates": [
                            {
                                "ticker": "CAT",
                                "cik": "1",
                                "company_name": "Caterpillar Inc.",
                                "rationale": "r",
                                "metrics_snapshot": {},
                            }
                        ],
                    }
                )
            )
        if spec.stage == "reader":
            return AgentResult(text=_draft().model_dump_json())
        if spec.stage == "critic":
            return AgentResult(
                text=json.dumps(
                    {
                        "ticker": "CAT",
                        "challenges": [
                            {
                                "target_claim_idx": 0,
                                "text": "c",
                                "severity": "material",
                                "citations": [
                                    {
                                        "source_ref": f"{ACC}#7¶1",
                                        "quote": "management flagged input cost pressure as a risk",
                                    }
                                ],
                            }
                        ]
                        * 2,
                        "overall_assessment": "thesis_weakened",
                        "what_would_change_my_mind": "x",
                    }
                )
            )
        # baseline or synthesizer -> a full memo
        return AgentResult(
            text=json.dumps(
                _memo().model_dump(
                    mode="json",
                    exclude={
                        "run_id",
                        "produced_by",
                        "created_at",
                        "token_cost",
                        "cost",
                        "disclaimer",
                        "schema_version",
                    },
                )
            )
        )

    runner = CallbackAgentRunner(responder)
    cmp = await compare.run_comparison(
        queries=["profitable industrials"], seeds=1, runner=runner, max_candidates=1
    )
    md = compare.render_markdown(cmp)
    assert "pipeline" in md and "baseline" in md
    assert "citation accuracy" in md
    assert {e.engine for e in cmp.engines} == {"pipeline", "baseline"}
    # Both engines produced at least one judged memo.
    assert all(e.n_memos >= 1 for e in cmp.engines)


async def test_compare_survives_one_failing_screen(mock_edgar):
    """A transient error on one screen must not abort the whole comparison (it did, twice)."""
    _seed()

    def responder(spec):
        # Simulate a transient SDK/transport failure on the poisoned query's screener call.
        if spec.stage == "screener" and "poison" in spec.prompt.lower():
            raise RuntimeError("Claude Code returned an error result: success")
        if spec.stage == "judge":
            return AgentResult(text=json.dumps({
                "grounding": 4, "argument_balance": 4, "specificity": 4,
                "readability": 4, "justification": "ok",
            }))
        if spec.stage == "screener":
            return AgentResult(text=json.dumps({
                "interpreted_intent": "i", "filters_applied": [],
                "candidates": [{"ticker": "CAT", "cik": "1", "company_name": "Caterpillar Inc.",
                                "rationale": "r", "metrics_snapshot": {}}],
            }))
        if spec.stage == "reader":
            return AgentResult(text=_draft().model_dump_json())
        if spec.stage == "critic":
            return AgentResult(text=json.dumps({
                "ticker": "CAT",
                "challenges": [{"target_claim_idx": 0, "text": "c", "severity": "material",
                                "citations": [{"source_ref": f"{ACC}#7¶1",
                                               "quote": "management flagged input cost pressure as a risk"}]}] * 2,
                "overall_assessment": "thesis_weakened", "what_would_change_my_mind": "x",
            }))
        return AgentResult(text=json.dumps(_memo().model_dump(mode="json", exclude={
            "run_id", "produced_by", "created_at", "token_cost", "cost", "disclaimer",
            "schema_version",
        })))

    runner = CallbackAgentRunner(responder)
    cmp = await compare.run_comparison(
        queries=["good industrials", "poison tech"],
        engines=["pipeline"],
        seeds=1,
        runner=runner,
        max_candidates=1,
    )
    # The comparison completed rather than raising, and the healthy screen still produced a memo.
    pipe = next(e for e in cmp.engines if e.engine == "pipeline")
    assert pipe.n_memos >= 1
    # Only the good query's run survived; the poisoned one was dropped after its retry.
    assert {r.query for r in cmp.run_metrics} == {"good industrials"}


async def test_compare_resume_skips_completed_screens(mock_edgar, tmp_path):
    """Re-invoking with the same eval_dir finishes only what's missing — a killed run wastes at
    most the in-flight screen, not the completed ones."""
    _seed()

    def responder(spec):
        if spec.stage == "judge":
            return AgentResult(text=json.dumps({
                "grounding": 4, "argument_balance": 4, "specificity": 4,
                "readability": 4, "justification": "ok",
            }))
        return AgentResult(text=json.dumps(_memo().model_dump(mode="json", exclude={
            "run_id", "produced_by", "created_at", "token_cost", "cost", "disclaimer",
            "schema_version",
        })))

    eval_dir = tmp_path / "cmp"
    r1 = CallbackAgentRunner(responder)
    cmp1 = await compare.run_comparison(
        queries=["profitable industrials"], engines=["baseline"], seeds=1,
        runner=r1, max_candidates=1, eval_dir=eval_dir,
    )
    assert len(r1.calls) > 0 and cmp1.engines[0].n_memos >= 1
    unit = eval_dir / "units"
    assert any(unit.glob("*.json"))  # the screen was persisted

    # Second invocation, same eval_dir, fresh runner: the completed screen is loaded from disk.
    r2 = CallbackAgentRunner(responder)
    cmp2 = await compare.run_comparison(
        queries=["profitable industrials"], engines=["baseline"], seeds=1,
        runner=r2, max_candidates=1, eval_dir=eval_dir,
    )
    assert len(r2.calls) == 0  # nothing re-run
    assert cmp2.engines[0].n_memos == cmp1.engines[0].n_memos  # identical assembled result
