"""Screener stage: NL request -> declarative filters -> shortlist of candidates."""

from __future__ import annotations

from desk.agents.base import AgentRunner, AgentStage, validate_tickers_in_universe
from desk.contracts.v1 import ScreenResult
from desk.orchestrator.run import RunContext
from desk.settings import load_yaml_config
from desk.tools import screen_tools

SCREENER_SYSTEM_PROMPT = """\
You are an equity screener. You convert a natural-language request into quantitative filters and
run them against a metrics universe. You NEVER invent tickers — only use tickers returned by the
`run_screen` tool.

Documented metric fields (the ONLY fields you may filter on):
  market_cap, trailing_pe, revenue_ttm, revenue_growth_yoy, gross_margin, operating_margin,
  gross_margin_trend, operating_margin_trend, net_debt_to_ebitda, sector (string, op "==").
Operators: "<", "<=", ">", ">=", "==", "between" (value is [low, high]).

Process:
1. Restate the request as `interpreted_intent`.
2. Translate it into declarative filters over the documented fields only. Reasonable defaults:
   "mid-cap" ~ market_cap between [2e9, 1e10]; "profitable" ~ operating_margin > 0;
   "improving margins" ~ operating_margin_trend > 0 (or gross_margin_trend > 0);
   "undervalued" ~ trailing_pe below a sector-reasonable threshold.
3. Call `run_screen` with your filters (sort by the most relevant field, limit <= 10).
4. If fewer than 2 rows match, relax the LEAST-central filter once and re-run. You MUST disclose
   every relaxation TWICE: as a structured entry in `relaxations` (the field, the original filter,
   the relaxed filter, and a short reason) AND in one sentence of `interpreted_intent`.
5. Choose 1-5 candidates. For each, write a <= 80-word rationale grounded in the returned metrics,
   and copy the ticker, cik, company_name, and a small metrics_snapshot from the row.

Output ONLY this JSON (no prose, no code fence):
{
  "interpreted_intent": str,
  "filters_applied": [ { "field": str, "op": str, "value": number | [number, number] } ],
  "relaxations": [ { "field": str,
                     "original": { "field": str, "op": str, "value": number | [number, number] },
                     "relaxed_to": { "field": str, "op": str, "value": number | [number, number] },
                     "reason": str } ],
  "candidates": [ { "ticker": str, "cik": str, "company_name": str, "rationale": str,
                    "metrics_snapshot": { "<field>": number|null } } ]
}
`relaxations` is empty when you did not loosen any filter. Do not produce buy/sell advice.
1 to 5 candidates.
"""


async def run_screener(
    query: str,
    *,
    run_id: str,
    runner: AgentRunner,
    max_candidates: int = 4,
    run_ctx: RunContext | None = None,
    degradations: list[str] | None = None,
    model: str | None = None,
    max_section_chars: int = 12_000,
    metrics_source=None,
    valid_tickers: set[str] | None = None,
    stage_name: str = "screener",
) -> ScreenResult:
    models = load_yaml_config("models").get("stages", {})
    budgets = load_yaml_config("budgets").get("stages", {}).get("screener", {})
    stage = AgentStage(
        name=stage_name,
        model=model or models.get("screener", "claude-haiku-4-5"),
        system_prompt=SCREENER_SYSTEM_PROMPT,
        runner=runner,
        run_id=run_id,
        allowed_tools=screen_tools.TOOL_NAMES,
        mcp_servers={"screen": screen_tools.build_server(metrics_source)},
        max_turns=int(budgets.get("max_turns", 10)) or 10,
        max_section_chars=max_section_chars,
        degradations=degradations or [],
    )

    # When running against a fixture universe (evals), validate candidates against the fixture's
    # tickers rather than the production universe.
    if valid_tickers is not None:
        allowed = {t.upper() for t in valid_tickers}

        def _ticker_validator(artifact) -> list[str]:
            bad = [c.ticker for c in artifact.candidates if c.ticker.upper() not in allowed]
            return [f"Ticker not in universe: {t!r}" for t in bad]

        validators = [_ticker_validator]
    else:
        validators = [validate_tickers_in_universe]
    prompt = (
        f"Screening request: {query}\n\n"
        f"Return {max_candidates} or fewer candidates as ScreenResult JSON."
    )
    if run_ctx is not None:
        run_ctx.write_prompt("screener", prompt)

    result = await stage.run(
        prompt,
        contract_cls=ScreenResult,
        semantic_validators=validators,
    )
    assert isinstance(result, ScreenResult)
    # Enforce the requested candidate cap deterministically.
    result.candidates = result.candidates[:max_candidates]
    return result
