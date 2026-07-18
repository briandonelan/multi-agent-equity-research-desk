"""Top-level orchestration: run the screener eval over a model matrix and collect evaluations.

This ties the runner (which produces raw artifacts) to the scoring (which turns them into numbers).
The CLI is a thin wrapper over :func:`evaluate_model` / :func:`run_matrix`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from desk.agents.base import AgentRunner
from desk.data.metrics_source import MetricsSource
from desk.evals.screener import runner as runner_mod
from desk.evals.screener.fixtures import fixture_tickers, load_fixture_source
from desk.evals.screener.scoring import ModelEvaluation, QueryEvaluation, evaluate_query
from desk.evals.screener.suite import SuiteQuery, load_dimension_map, load_suite

# Short aliases the CLI accepts for --models. "default" (or an empty spec) runs each stage's own
# configured model, so it costs the least and reflects the shipped screener.
MODEL_ALIASES: dict[str, str | None] = {
    "default": None,
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-4-8",
}


def resolve_models(spec: str | None) -> list[str | None]:
    """Turn a ``--models haiku,sonnet`` spec into a list of model strings (or None for default)."""
    if not spec:
        return [None]
    out: list[str | None] = []
    for raw in spec.split(","):
        name = raw.strip()
        if not name:
            continue
        # Accept both an alias and a full model id verbatim.
        out.append(MODEL_ALIASES.get(name, name))
    return out or [None]


DEFAULT_CONCURRENCY = 4


async def evaluate_model(
    model: str | None,
    queries: list[SuiteQuery],
    *,
    runner: AgentRunner,
    run_id: str,
    repeats: int = 2,
    dimension_map: dict[str, list[str]] | None = None,
    metrics_source: MetricsSource | None = None,
    valid_tickers: set[str] | None = None,
    semaphore: asyncio.Semaphore | None = None,
    progress: Callable[[str], None] | None = None,
) -> ModelEvaluation:
    dim_map = dimension_map or load_dimension_map()
    source = metrics_source or load_fixture_source()
    tickers = valid_tickers or fixture_tickers()
    label = model or "default"
    sem = semaphore or asyncio.Semaphore(DEFAULT_CONCURRENCY)

    async def _one(q: SuiteQuery) -> QueryEvaluation:
        obs = await runner_mod.observe_query(
            q,
            runner=runner,
            run_id=run_id,
            model=model,
            metrics_source=source,
            valid_tickers=tickers,
            repeats=repeats,
            semaphore=sem,
        )
        ev = evaluate_query(q, obs, dim_map)
        if progress:
            progress(f"[{label}] {q.id} done")
        return ev

    # Queries run concurrently; the shared semaphore bounds total in-flight screener calls.
    # gather preserves order, so the report's query order stays stable across runs.
    results = await asyncio.gather(*[_one(q) for q in queries])
    return ModelEvaluation(model=label, queries=list(results))


async def run_matrix(
    models: list[str | None],
    queries: list[SuiteQuery] | None = None,
    *,
    runner: AgentRunner,
    run_id: str,
    repeats: int = 2,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: Callable[[str], None] | None = None,
) -> list[ModelEvaluation]:
    """Evaluate every model on the same queries. ``models=[None]`` uses each stage's default.

    A single semaphore is shared across all models and queries, so ``concurrency`` caps the total
    number of live screener calls regardless of how the matrix is shaped.
    """
    qs = queries if queries is not None else load_suite()
    dim_map = load_dimension_map()
    source = load_fixture_source()
    tickers = fixture_tickers()
    sem = asyncio.Semaphore(max(1, concurrency))
    out: list[ModelEvaluation] = []
    for model in models:
        out.append(
            await evaluate_model(
                model,
                qs,
                runner=runner,
                run_id=run_id,
                repeats=repeats,
                dimension_map=dim_map,
                metrics_source=source,
                valid_tickers=tickers,
                semaphore=sem,
                progress=progress,
            )
        )
    return out
