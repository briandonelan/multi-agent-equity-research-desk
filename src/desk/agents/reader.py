"""Fundamentals Reader stage: for one ticker, draft a cited bull case."""

from __future__ import annotations

from desk.agents.base import (
    AgentRunner,
    AgentStage,
    validate_citations,
    validate_tickers_in_universe,
)
from desk.contracts.v1 import Candidate, ThesisDraft
from desk.orchestrator.run import RunContext
from desk.settings import load_yaml_config
from desk.tools import filing_tools

READER_SYSTEM_PROMPT = """\
You are a fundamentals analyst. For a SINGLE company you read its recent SEC filings and draft an
evidence-grounded bull case. You are strictly evidence-first.

Tools (filings only): `list_filings`, `get_section` (10-K/10-Q items with citeable source_refs),
`get_xbrl_facts` (numeric facts with period labels).

Process:
1. `list_filings` for the ticker. Prefer the most recent 10-K (Item 1A Risk Factors, Item 7 MD&A)
   and the latest 10-Q (Item 2 MD&A).
2. `get_section` on those items. Every claim you make MUST cite >= 1 `source_ref` returned by
   `get_section`, with a SHORT verbatim quote (<= 40 words) copied exactly from that passage.
3. Numbers MUST come from `get_xbrl_facts`, never from memory.
4. Label each claim's `claim_type` honestly: "fact", "interpretation", or "projection".
   Projections must be MANAGEMENT'S (cited), not your own forecasts.
5. Write a <= 120-word `thesis_summary` and 4-8 claims.

Output ONLY this JSON (no prose, no code fence):
{
  "ticker": str,
  "thesis_summary": str,
  "claims": [ { "text": str, "claim_type": "fact"|"interpretation"|"projection",
                "citations": [ { "source_ref": str, "quote": str } ] } ]
}
source_ref values MUST be copied exactly from a `get_section` result; quotes MUST be verbatim.
No buy/sell advice, no price targets. 4 to 8 claims, each with >= 1 citation.
"""


async def run_reader(
    candidate: Candidate,
    *,
    run_id: str,
    runner: AgentRunner,
    run_ctx: RunContext | None = None,
    degradations: list[str] | None = None,
    model: str | None = None,
    max_section_chars: int = 12_000,
    max_filings: int | None = None,
) -> ThesisDraft:
    models = load_yaml_config("models").get("stages", {})
    budgets = load_yaml_config("budgets").get("stages", {}).get("reader", {})
    stage = AgentStage(
        name="reader",
        model=model or models.get("reader", "claude-sonnet-5"),
        system_prompt=READER_SYSTEM_PROMPT,
        runner=runner,
        run_id=run_id,
        allowed_tools=filing_tools.TOOL_NAMES,
        mcp_servers={"filings": filing_tools.build_server()},
        max_turns=int(budgets.get("max_turns", 16)) or 16,
        max_section_chars=max_section_chars,
        degradations=degradations or [],
    )
    budget_note = (
        f"\nBUDGET CONSTRAINT: read only the {max_filings} most recent filings.\n"
        if max_filings
        else ""
    )
    prompt = (
        f"Draft a bull case for {candidate.ticker} ({candidate.company_name}).\n"
        f"Screener rationale: {candidate.rationale}\n"
        f"{budget_note}\n"
        f"Return the ThesisDraft JSON."
    )
    if run_ctx is not None:
        run_ctx.write_prompt(f"reader_{candidate.ticker}", prompt)

    draft = await stage.run(
        prompt,
        contract_cls=ThesisDraft,
        ticker=candidate.ticker,
        semantic_validators=[validate_citations, validate_tickers_in_universe],
    )
    assert isinstance(draft, ThesisDraft)
    return draft
