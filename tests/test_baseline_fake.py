"""Baseline engine end-to-end with FakeAgentRunner: validation, repair-retry, HandoffFailure."""

from __future__ import annotations

import json

import pytest

from desk.agents.base import AgentResult, FakeAgentRunner
from desk.agents.baseline import run_baseline
from desk.contracts.v1 import HandoffFailure
from desk.data import sections
from desk.orchestrator.pipeline import run_memo
from desk.orchestrator.render import render_memo


def _seed_cat_section() -> str:
    text = (
        "Revenue increased 4 percent year over year driven by higher machine volumes."
        "\n\n"
        "Operating margin was 16 percent, and management flagged input cost pressure as a risk."
    )
    sections.store_section_text("0000000021-25-000001", "7", text)
    return "0000000021-25-000001#7¶0"


def _valid_memo_json(source_ref: str) -> dict:
    cite = {"source_ref": source_ref, "quote": "Revenue increased 4 percent year over year"}
    return {
        "ticker": "CAT",
        "company_name": "Caterpillar Inc.",
        "thesis_summary": "Cyclical industrial with steady volume growth.",
        "valuation_snapshot": {"market_cap": 4.2e11, "operating_margin": 0.16},
        "bull_case": [
            {"text": "Revenue is growing.", "claim_type": "fact", "citations": [cite]},
            {"text": "Volumes rose.", "claim_type": "interpretation", "citations": [cite]},
        ],
        "bear_case": [
            {
                "target_claim_idx": 0,
                "text": "Input costs pressure margins.",
                "severity": "material",
                "citations": [
                    {
                        # This quote lives in passage ¶1, so its source_ref must point there.
                        "source_ref": source_ref.replace("¶0", "¶1"),
                        "quote": "management flagged input cost pressure as a risk",
                    }
                ],
            },
            {
                "target_claim_idx": None,
                "text": "Cyclicality risk.",
                "severity": "minor",
                "citations": [],
            },
        ],
        "unresolved_disagreements": ["Whether margin pressure is transitory."],
        "confidence": "medium",
        "confidence_rationale": "One material challenge, otherwise grounded.",
    }


async def test_baseline_produces_valid_memo(mock_edgar):
    ref = _seed_cat_section()
    runner = FakeAgentRunner(
        [
            AgentResult(
                text=json.dumps(_valid_memo_json(ref)),
                usage={"input_tokens": 5000, "output_tokens": 800},
            )
        ]
    )
    memos = await run_baseline("industrials with growth", run_id="RUNA", runner=runner)
    assert len(memos) == 1
    memo = memos[0]
    assert memo.ticker == "CAT"
    assert memo.disclaimer  # injected
    assert memo.bull_case and memo.bear_case
    # Cost was filled from the ledger (one logged call).
    assert memo.cost.n_calls == 1
    # Rendered markdown carries the disclaimer and a cost line.
    md = render_memo(memo)
    assert "investment advice" in md.lower()
    assert "This memo cost" in md


async def test_repair_retry_recovers(mock_edgar):
    ref = _seed_cat_section()
    bad = json.dumps({"ticker": "CAT"})  # missing required fields
    good = json.dumps(_valid_memo_json(ref))
    runner = FakeAgentRunner([AgentResult(text=bad), AgentResult(text=good)])
    memos = await run_baseline("x", run_id="RUNB", runner=runner)
    assert len(memos) == 1
    assert len(runner.calls) == 2  # first attempt + one repair retry


async def test_two_failures_raise_handoff_failure(mock_edgar):
    runner = FakeAgentRunner(
        [AgentResult(text='{"ticker":"CAT"}'), AgentResult(text='{"ticker":"CAT"}')]
    )
    with pytest.raises(HandoffFailure) as exc:
        await run_baseline("x", run_id="RUNC", runner=runner)
    assert exc.value.stage == "baseline"
    assert exc.value.errors


async def test_run_memo_persists_and_handles_failure(mock_edgar):
    # A run whose only agent fails should not crash: it records the failure and yields no memo.
    runner = FakeAgentRunner([AgentResult(text="garbage"), AgentResult(text="garbage")])
    result = await run_memo("x", engine="baseline", runner=runner, max_candidates=1)
    assert result.memos == []
    assert result.failures
