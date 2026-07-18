"""Live smoke test — hits the real Claude Agent SDK + SEC. Skipped in CI.

Run explicitly with:  uv run pytest -m live
Requires ANTHROPIC (an authenticated `claude` CLI session or ANTHROPIC_API_KEY) + SEC_EDGAR_USER_AGENT.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


async def test_live_baseline_memo(tmp_path, monkeypatch):
    if not os.environ.get("SEC_EDGAR_USER_AGENT"):
        pytest.skip("SEC_EDGAR_USER_AGENT not set")
    # Use the real cache so the pre-built universe metrics are available.
    from desk.orchestrator.pipeline import run_memo

    result = await run_memo(
        "profitable large-cap industrials with improving margins",
        engine="baseline",
        max_candidates=1,
    )
    assert result.memos, f"no memo produced; failures={result.failures}"
    memo = result.memos[0]
    assert memo.bull_case, "expected a bull case"
    assert memo.cost.total_cost_usd > 0
    # Citations were validated at the boundary; confirm they're present and resolvable-shaped.
    for claim in memo.bull_case:
        assert claim.citations
        assert "¶" in claim.citations[0].source_ref
