"""Execute the real screener agent against the fixture universe.

The screener is the *unit under test*: it sees only the synthetic fixture (via an injected
:class:`MetricsSource`) and validates its tickers against the fixture universe, so no live data or
production code path is touched. This module produces the raw :class:`ScreenResult` artifacts; the
scoring/report modules turn them into numbers. Every call is tagged ``stage="screener-eval"`` in
the ledger so eval spend is separable from production runs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from desk.agents import screener as screener_agent
from desk.agents.base import AgentRunner
from desk.contracts.v1 import HandoffFailure, ScreenResult
from desk.data.metrics_source import MetricsSource
from desk.evals.screener.fixtures import fixture_tickers, load_fixture_source
from desk.evals.screener.suite import SuiteQuery

EVAL_STAGE = "screener-eval"


@dataclass
class SeedObservation:
    """Everything one seed of one query produced: base run plus its variants."""

    seed: int
    base: ScreenResult | None = None
    paraphrases: list[ScreenResult | None] = field(default_factory=list)
    perturbations: list[ScreenResult | None] = field(default_factory=list)
    error: str | None = None


async def _run_one(
    text: str,
    *,
    runner: AgentRunner,
    model: str | None,
    metrics_source: MetricsSource,
    valid_tickers: set[str],
    run_id: str,
    max_candidates: int,
) -> ScreenResult:
    return await screener_agent.run_screener(
        text,
        run_id=run_id,
        runner=runner,
        model=model,
        max_candidates=max_candidates,
        metrics_source=metrics_source,
        valid_tickers=valid_tickers,
        stage_name=EVAL_STAGE,
    )


async def observe_query(
    q: SuiteQuery,
    *,
    runner: AgentRunner,
    run_id: str,
    model: str | None = None,
    metrics_source: MetricsSource | None = None,
    valid_tickers: set[str] | None = None,
    repeats: int = 2,
    max_candidates: int = 5,
    include_paraphrases: bool = True,
    include_perturbations: bool = True,
    semaphore: asyncio.Semaphore | None = None,
) -> list[SeedObservation]:
    """Run the screener on the query's base text and every variant, ``repeats`` times.

    The base, its paraphrases, and its perturbations are independent LLM calls, so they run
    concurrently; ``semaphore`` (shared across the whole matrix) bounds how many are in flight at
    once so we don't trip API rate limits. A variant that fails handoff validation is recorded as
    ``None`` rather than aborting the run — one flaky translation should not lose the query's other
    numbers. A seed whose *base* run fails is marked errored, since the base is the reference the
    other checks compare against.
    """
    source = metrics_source or load_fixture_source()
    tickers = valid_tickers or fixture_tickers()

    async def _call(text: str) -> ScreenResult:
        if semaphore is None:
            return await _run_one(
                text, runner=runner, model=model, metrics_source=source,
                valid_tickers=tickers, run_id=run_id, max_candidates=max_candidates,
            )
        async with semaphore:
            return await _run_one(
                text, runner=runner, model=model, metrics_source=source,
                valid_tickers=tickers, run_id=run_id, max_candidates=max_candidates,
            )

    async def _safe_call(text: str) -> tuple[ScreenResult | None, str | None]:
        # Eval resilience: a single failed call (handoff rejection, max-turns, transport error)
        # must not abort the whole matrix — record it and let the other calls proceed. This is the
        # same "hold the harness to the pipeline's standard" discipline the desk itself uses.
        try:
            return (await _call(text), None)
        except HandoffFailure as e:
            return (None, f"handoff: {e.errors}")
        except Exception as e:  # noqa: BLE001 — intentionally broad; the eval never crashes on one call
            return (None, f"{type(e).__name__}: {e}")

    async def _observe_seed(seed: int) -> SeedObservation:
        obs = SeedObservation(seed=seed)
        base_task = asyncio.ensure_future(_safe_call(q.text))
        para_tasks = (
            [asyncio.ensure_future(_safe_call(p)) for p in q.paraphrases]
            if include_paraphrases
            else []
        )
        pert_tasks = (
            [asyncio.ensure_future(_safe_call(pt.text)) for pt in q.perturbations]
            if include_perturbations
            else []
        )
        base_res, base_err = await base_task
        obs.base = base_res
        obs.paraphrases = [r for r, _ in [await t for t in para_tasks]]
        obs.perturbations = [r for r, _ in [await t for t in pert_tasks]]
        if obs.base is None:
            obs.error = base_err or "base run failed"
        return obs

    seeds = await asyncio.gather(*[_observe_seed(s) for s in range(max(1, repeats))])
    return list(seeds)
