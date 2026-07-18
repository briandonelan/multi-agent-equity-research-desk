"""Contrarian Critic stage.

The critic attacks the bull case adversarially WITHOUT seeing the reader's chain of reasoning.
Isolation is the sycophancy control: in ``contract`` handoff mode the critic receives only the
claims (via :meth:`ThesisDraft.for_critic`) — no thesis summary, no confidence language. In
``full_context`` mode it additionally receives the reader's thesis summary. It must hunt for disconfirming evidence in the SAME filings and cite it.
"""

from __future__ import annotations

import json

from desk.agents.base import (
    AgentRunner,
    AgentStage,
    validate_citations,
    validate_tickers_in_universe,
)
from desk.contracts.v1 import CritiqueReport, ThesisDraft
from desk.orchestrator.run import RunContext
from desk.settings import load_yaml_config
from desk.tools import filing_tools

CRITIC_SYSTEM_PROMPT = """\
You are a contrarian equity critic. You are given a set of numbered bull-case CLAIMS (text +
citations) for one company. Your job is to attack them adversarially using the SAME SEC filings,
hunting for disconfirming evidence. You do NOT see the analyst's reasoning or confidence — judge
the claims on the evidence alone.

Tools (filings only): `list_filings`, `get_section`, `get_xbrl_facts`. Use them to verify claims
and to find contradictions. Every challenge you raise MUST cite >= 1 `source_ref` (with a short
verbatim quote) that you obtained from `get_section` this session.

You MUST check, at minimum:
- fiscal-period consistency (is a claimed trend real across periods, or a single-quarter blip?),
- one-off items dressed up as recurring trends (disposition gains, tax items, restructuring),
- debt / liquidity (net debt, interest expense, maturities),
- whether the Risk Factors (Item 1A) contradict MD&A optimism.

Assign each challenge a severity honestly: "minor", "material", or "fatal". Set
`target_claim_idx` to the 0-based index of the claim you challenge, or null if you challenge the
thesis as a whole. Agreeing with a claim is acceptable ONLY with documented verification steps —
but you must still surface the strongest genuine challenges you can find.

`what_would_change_my_mind` is MANDATORY: state the specific evidence that would resolve your
strongest challenge.

Output ONLY this JSON (no prose, no code fence):
{
  "ticker": str,
  "challenges": [ { "target_claim_idx": int|null, "text": str,
                    "severity": "minor"|"material"|"fatal",
                    "citations": [ { "source_ref": str, "quote": str } ] } ],
  "overall_assessment": "thesis_holds"|"thesis_weakened"|"thesis_rejected",
  "what_would_change_my_mind": str
}
2 to 6 challenges. source_ref/quote must be real (from `get_section`). No buy/sell advice.
"""


def _claims_payload(thesis: ThesisDraft, handoff_mode: str) -> dict:
    """What the critic sees. contract -> claims only (isolation ON); full_context -> + summary."""
    view = thesis.for_critic()
    payload: dict = {
        "ticker": view.ticker,
        "claims": [
            {
                "index": i,
                "text": c.text,
                "claim_type": c.claim_type,
                "citations": [
                    {"source_ref": ci.source_ref, "quote": ci.quote} for ci in c.citations
                ],
            }
            for i, c in enumerate(view.claims)
        ],
    }
    if handoff_mode == "full_context":
        payload["analyst_thesis_summary"] = thesis.thesis_summary
    return payload


async def run_critic(
    thesis: ThesisDraft,
    *,
    run_id: str,
    runner: AgentRunner,
    handoff_mode: str = "contract",
    run_ctx: RunContext | None = None,
    degradations: list[str] | None = None,
    model: str | None = None,
    max_section_chars: int = 12_000,
) -> CritiqueReport:
    models = load_yaml_config("models").get("stages", {})
    budgets = load_yaml_config("budgets").get("stages", {}).get("critic", {})
    stage = AgentStage(
        name="critic",
        model=model or models.get("critic", "claude-opus-4-8"),
        system_prompt=CRITIC_SYSTEM_PROMPT,
        runner=runner,
        run_id=run_id,
        allowed_tools=filing_tools.TOOL_NAMES,
        mcp_servers={"filings": filing_tools.build_server()},
        max_turns=int(budgets.get("max_turns", 16)) or 16,
        max_section_chars=max_section_chars,
        degradations=degradations or [],
    )
    payload = _claims_payload(thesis, handoff_mode)
    prompt = (
        f"Company: {thesis.ticker}. Attack the following bull-case claims adversarially, using "
        f"the filings. Claims:\n\n{json.dumps(payload, indent=2)}\n\nReturn the CritiqueReport JSON."
    )
    if run_ctx is not None:
        run_ctx.write_prompt(f"critic_{thesis.ticker}", prompt)

    report = await stage.run(
        prompt,
        contract_cls=CritiqueReport,
        ticker=thesis.ticker,
        semantic_validators=[validate_citations, validate_tickers_in_universe],
    )
    assert isinstance(report, CritiqueReport)
    return report
