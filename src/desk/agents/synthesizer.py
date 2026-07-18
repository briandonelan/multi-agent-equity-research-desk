"""Synthesizer stage: merge thesis + critique into the final memo.

No new claims. Materially/fatally challenged claims are MOVED into unresolved_disagreements
rather than deleted; confidence must cite the balance of severities. Preserves disagreement.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from desk.agents.base import (
    AgentRunner,
    AgentStage,
    validate_citations,
    validate_tickers_in_universe,
)
from desk.contracts.v1 import (
    DISCLAIMER,
    Candidate,
    CritiqueReport,
    ResearchMemo,
    RunCostSummary,
    ThesisDraft,
)
from desk.ledger import report
from desk.orchestrator.run import RunContext
from desk.settings import load_yaml_config

SYNTHESIZER_SYSTEM_PROMPT = """\
You are a research editor. You merge an analyst's bull-case THESIS and a critic's CRITIQUE into a
single balanced memo. You have NO tools and you introduce NO new claims or citations.

Rules:
- The bull_case is drawn from the thesis claims. You MAY drop a weak claim, but any claim that the
  critique challenges at "material" or "fatal" severity MUST be moved into
  `unresolved_disagreements` (a short plain-English sentence) rather than silently deleted. You may
  keep it in the bull_case too, but the disagreement must be surfaced.
- The bear_case is drawn from the critique challenges. Copy them faithfully.
- Do NOT alter any source_ref or quote — copy them EXACTLY from the thesis/critique you are given.
- `confidence` (low|medium|high) must reflect the balance of challenge severities: many
  material/fatal challenges -> lower confidence. Justify in `confidence_rationale`.
- If the critique is marked UNAVAILABLE, set bear_case to [] and add an explicit
  unresolved_disagreements entry stating that adversarial critique was unavailable for this memo.
- No buy/sell/hold advice, no price targets.

Output ONLY this JSON (no prose, no code fence):
{
  "ticker": str, "company_name": str,
  "thesis_summary": str,
  "valuation_snapshot": { "<metric>": number|null },
  "bull_case": [ { "text": str, "claim_type": "fact"|"interpretation"|"projection",
                   "citations": [ { "source_ref": str, "quote": str } ] } ],
  "bear_case": [ { "target_claim_idx": int|null, "text": str,
                   "severity": "minor"|"material"|"fatal",
                   "citations": [ { "source_ref": str, "quote": str } ] } ],
  "unresolved_disagreements": [ str ],
  "confidence": "low"|"medium"|"high",
  "confidence_rationale": str
}
"""


async def run_synthesizer(
    candidate: Candidate,
    thesis: ThesisDraft,
    critique: CritiqueReport | None,
    *,
    run_id: str,
    runner: AgentRunner,
    run_ctx: RunContext | None = None,
    degradations: list[str] | None = None,
    model: str | None = None,
) -> ResearchMemo:
    models = load_yaml_config("models").get("stages", {})
    budgets = load_yaml_config("budgets").get("stages", {}).get("synthesizer", {})
    stage = AgentStage(
        name="synthesizer",
        model=model or models.get("synthesizer", "claude-sonnet-5"),
        system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
        runner=runner,
        run_id=run_id,
        allowed_tools=[],
        mcp_servers={},
        max_turns=int(budgets.get("max_turns", 4)) or 4,
        degradations=degradations or [],
    )

    critique_payload = (
        critique.model_dump(
            mode="json", include={"challenges", "overall_assessment", "what_would_change_my_mind"}
        )
        if critique is not None
        else "UNAVAILABLE"
    )
    thesis_payload = thesis.model_dump(mode="json", include={"thesis_summary", "claims"})
    prompt = (
        f"Company: {candidate.ticker} ({candidate.company_name}).\n"
        f"Valuation metrics (for valuation_snapshot): {json.dumps(candidate.metrics_snapshot)}\n\n"
        f"THESIS:\n{json.dumps(thesis_payload, indent=2)}\n\n"
        f"CRITIQUE:\n{json.dumps(critique_payload, indent=2) if critique else 'UNAVAILABLE'}\n\n"
        f"Return the ResearchMemo JSON."
    )
    if run_ctx is not None:
        run_ctx.write_prompt(f"synthesizer_{candidate.ticker}", prompt)

    inject = {
        "cost": RunCostSummary(run_id=run_id).model_dump(),
        "disclaimer": DISCLAIMER,
        "as_of": datetime.now(UTC).date().isoformat(),
        "company_name": candidate.company_name,
    }
    memo = await stage.run(
        prompt,
        contract_cls=ResearchMemo,
        ticker=candidate.ticker,
        semantic_validators=[validate_citations, validate_tickers_in_universe],
        inject=inject,
    )
    assert isinstance(memo, ResearchMemo)
    memo.cost = report.build_cost_summary(run_id)
    return memo
