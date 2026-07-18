"""Single-agent baseline control.

One Sonnet-tier agent with BOTH MCP tool servers and one big prompt asking for the full
``ResearchMemo`` directly. This is the honest control the multi-agent pipeline is measured
against — it must exist and work before the pipeline is tuned.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from desk.agents.base import (
    AgentRunner,
    AgentStage,
    validate_citations,
    validate_tickers_in_universe,
)
from desk.contracts.v1 import DISCLAIMER, ResearchMemo, RunCostSummary
from desk.ledger import report
from desk.orchestrator.run import RunContext
from desk.settings import load_yaml_config
from desk.tools import filing_tools, screen_tools

BASELINE_SYSTEM_PROMPT = """\
You are an equity research analyst producing a single, evidence-grounded research memo from SEC
filings. You are the sole agent: you screen, read, AND critique.

You have two tool servers:
- screen tools: `run_screen` (declarative filters over a metrics universe) and `get_metrics`.
- filing tools: `list_filings`, `get_section` (10-K/10-Q items with citeable source_refs), and
  `get_xbrl_facts` (numeric XBRL facts with period labels).

Process:
1. Restate the request as declarative filters over DOCUMENTED metric fields only (market_cap,
   trailing_pe, revenue_ttm, revenue_growth_yoy, gross_margin, operating_margin,
   gross_margin_trend, operating_margin_trend, net_debt_to_ebitda, sector). Call `run_screen`.
   Never invent tickers — only use tickers returned by the tools.
2. Pick the SINGLE best-matching in-universe company.
3. Use `list_filings` then `get_section` (prefer the most recent 10-K's Risk Factors [1A] and
   MD&A [7], plus the latest 10-Q's MD&A [2]) and `get_xbrl_facts` for numbers. Every number
   must come from `get_xbrl_facts`, never from memory.
4. Build a BULL case: 4-8 claims. Each claim MUST cite >= 1 source_ref returned by `get_section`,
   with a short verbatim quote (<= 40 words) copied from that passage. Label each claim's type
   honestly: "fact", "interpretation", or "projection" (projections must be management's, cited).
5. Build a BEAR case adversarially from the SAME filings: 2-6 challenges, each citing sources.
   Check fiscal-period consistency, one-off items dressed as trends, debt/liquidity, and whether
   risk factors contradict MD&A optimism. Assign severity: "minor", "material", or "fatal".
6. Record any unresolved disagreements explicitly — never silently reconcile bull and bear.

Output ONLY a single JSON object matching this ResearchMemo schema (no prose, no code fence):
{
  "ticker": str, "company_name": str,
  "thesis_summary": str,
  "valuation_snapshot": { "market_cap": number|null, "trailing_pe": number|null,
                          "revenue_ttm": number|null, "operating_margin": number|null },
  "bull_case": [ { "text": str, "claim_type": "fact"|"interpretation"|"projection",
                   "citations": [ { "source_ref": str, "quote": str } ] } ],
  "bear_case": [ { "target_claim_idx": int|null, "text": str,
                   "severity": "minor"|"material"|"fatal",
                   "citations": [ { "source_ref": str, "quote": str } ] } ],
  "unresolved_disagreements": [ str ],
  "confidence": "low"|"medium"|"high",
  "confidence_rationale": str
}
source_ref values MUST be copied exactly from a `get_section` result; quotes MUST be verbatim
from that passage. Do not include buy/sell/hold advice or price targets.
"""


def _placeholder_cost(run_id: str) -> dict:
    return RunCostSummary(run_id=run_id).model_dump()


async def run_baseline(
    query: str,
    *,
    run_id: str,
    runner: AgentRunner,
    run_ctx: RunContext | None = None,
    max_candidates: int = 1,
) -> list[ResearchMemo]:
    """Run the baseline engine and return the produced memo(s) (currently one)."""
    models = load_yaml_config("models").get("stages", {})
    budgets = load_yaml_config("budgets")
    model = models.get("baseline", "claude-sonnet-5")
    max_turns = int(budgets.get("stages", {}).get("baseline", {}).get("max_turns", 24)) or 24

    screen_server = screen_tools.build_server()
    filing_server = filing_tools.build_server()

    stage = AgentStage(
        name="baseline",
        model=model,
        system_prompt=BASELINE_SYSTEM_PROMPT,
        runner=runner,
        run_id=run_id,
        allowed_tools=screen_tools.TOOL_NAMES + filing_tools.TOOL_NAMES,
        mcp_servers={"screen": screen_server, "filings": filing_server},
        max_turns=max_turns,
    )

    prompt = (
        f"Screening request: {query}\n\n"
        f"Produce a research memo for the single best-matching in-universe company, following "
        f"your process. Return only the ResearchMemo JSON."
    )
    if run_ctx is not None:
        run_ctx.write_prompt("baseline", prompt)

    today = datetime.now(UTC).date()
    memo = await stage.run(
        prompt,
        contract_cls=ResearchMemo,
        semantic_validators=[validate_citations, validate_tickers_in_universe],
        inject={
            "cost": _placeholder_cost(run_id),
            "disclaimer": DISCLAIMER,
            "as_of": today.isoformat(),
        },
    )
    assert isinstance(memo, ResearchMemo)

    # Replace the placeholder cost with the real per-run roll-up now that calls are logged.
    memo.cost = report.build_cost_summary(run_id)
    return [memo]


def today() -> date:
    return datetime.now(UTC).date()
