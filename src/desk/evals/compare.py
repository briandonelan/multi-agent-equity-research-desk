"""Baseline vs pipeline comparison — the project's centerpiece artifact.

Runs a fixed set of committed screening queries through both engines, judges every memo (LLM +
programmatic citation accuracy), and emits a Markdown table: judge scores, citation accuracy,
cost/memo, tokens/memo, wall-clock, and failure counts. ``--json`` exports the raw rows for the
blog's cost/quality section.

Live cost scales with (queries x engines x seeds); the CLI defaults are small and callers opt
into scale explicitly.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

from desk.agents.base import AgentRunner, SdkAgentRunner
from desk.evals.judge import judge_memo
from desk.ledger import report
from desk.orchestrator.pipeline import run_memo

# Committed screening queries — the fixed evaluation set. The first eight were the original
# comparison; four more were added to widen sector coverage (consumer and financials, which the
# original set never touched) for a larger, more diverse scoreboard.
COMMITTED_QUERIES = [
    "profitable mid-cap industrials with improving margins",
    "large-cap technology companies with strong revenue growth",
    "industrials with low leverage and expanding operating margins",
    "defensive healthcare names with steady revenue",
    "diversified industrials trading at a reasonable P/E",
    "high-margin software or semiconductor businesses",
    "energy or materials companies with improving profitability",
    "aerospace and defense contractors with growing backlogs",
    "mega-cap technology with the widest operating margins",
    "large pharmaceutical companies with wide gross margins",
    "consumer companies with durable brands and pricing power",
    "money-center banks trading below the market's earnings multiple",
]


@dataclass
class MemoMetric:
    query: str
    engine: str
    ticker: str
    citation_accuracy: float
    grounding: int
    argument_balance: int
    specificity: int
    readability: int


@dataclass
class RunMetric:
    query: str
    engine: str
    run_id: str
    n_memos: int
    n_failures: int
    cost_usd: float
    total_tokens: int
    wall_clock_s: float
    n_timeouts: int = 0  # subset of n_failures that were wall-clock timeouts (default: back-compat)


@dataclass
class EngineSummary:
    engine: str
    n_runs: int = 0
    n_memos: int = 0
    n_failures: int = 0
    n_timeouts: int = 0
    citation_accuracy: float = 0.0
    grounding: float = 0.0
    argument_balance: float = 0.0
    specificity: float = 0.0
    readability: float = 0.0
    cost_per_memo: float = 0.0
    tokens_per_memo: float = 0.0
    wall_clock_s: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__


@dataclass
class Comparison:
    engines: list[EngineSummary] = field(default_factory=list)
    memo_metrics: list[MemoMetric] = field(default_factory=list)
    run_metrics: list[RunMetric] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "engines": [e.as_dict() for e in self.engines],
            "memo_metrics": [m.__dict__ for m in self.memo_metrics],
            "run_metrics": [r.__dict__ for r in self.run_metrics],
        }


def _unit_path(eval_dir: Path, query: str, engine: str, seed: int) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:50]
    return eval_dir / "units" / f"{slug}__{engine}__s{seed}.json"


def _save_unit(path: Path, rm: RunMetric, mms: list[MemoMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"run_metric": asdict(rm), "memo_metrics": [asdict(m) for m in mms]}, indent=2),
        "utf-8",
    )


def _load_unit(path: Path) -> tuple[RunMetric, list[MemoMetric]]:
    d = json.loads(path.read_text("utf-8"))
    return (
        RunMetric(**d["run_metric"]),
        [MemoMetric(**m) for m in d["memo_metrics"]],
    )


async def run_comparison(
    *,
    queries: list[str] | None = None,
    engines: list[str] | None = None,
    seeds: int = 1,
    runner: AgentRunner | None = None,
    max_candidates: int = 3,
    concurrency: int = 1,
    progress=None,
    eval_dir: Path | None = None,
    resume: bool = True,
) -> Comparison:
    """Run every (query, engine, seed) screen and summarize.

    When ``eval_dir`` is given, each completed screen's metrics are written to
    ``eval_dir/units/`` as soon as it finishes, and (with ``resume``) a screen whose metrics
    already exist is skipped. This makes the comparison resumable: a killed run loses at most the
    screens that were in flight, and re-invoking with the same ``eval_dir`` finishes only what's
    missing.
    """
    queries = queries or COMMITTED_QUERIES
    engines = engines or ["pipeline", "baseline"]
    runner = runner or SdkAgentRunner()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _run_once(query: str, engine: str) -> tuple[RunMetric, list[MemoMetric]]:
        started = time.monotonic()
        result = await run_memo(
            query, engine=engine, max_candidates=max_candidates, runner=runner
        )
        elapsed = time.monotonic() - started
        roll = report.run_rollup(result.run_id)
        rm = RunMetric(
            query=query,
            engine=engine,
            run_id=result.run_id,
            n_memos=len(result.memos),
            n_failures=len(result.failures),
            n_timeouts=sum(1 for f in result.failures if f.get("timeout")),
            cost_usd=roll.total_cost_usd,
            total_tokens=roll.total_input_tokens + roll.total_output_tokens,
            wall_clock_s=round(elapsed, 2),
        )
        mms: list[MemoMetric] = []
        for memo in result.memos:
            score = await judge_memo(memo, run_id=result.run_id, runner=runner)
            mms.append(
                MemoMetric(
                    query=query,
                    engine=engine,
                    ticker=memo.ticker,
                    citation_accuracy=score.citation_accuracy,
                    grounding=score.grounding,
                    argument_balance=score.argument_balance,
                    specificity=score.specificity,
                    readability=score.readability,
                )
            )
        return rm, mms

    async def _unit(query: str, engine: str, seed: int) -> tuple[RunMetric, list[MemoMetric]] | None:
        # One (query, engine, seed) screen. The semaphore bounds how many full screens run at
        # once. Note: with concurrency > 1, per-run wall_clock includes contention and is not a
        # clean latency measurement — cost/tokens (from the ledger) stay exact. Resilience: a
        # transient SDK/transport error is retried once, then the screen is dropped and logged
        # rather than aborting the whole comparison. One flaky call must not cost 23 good screens.
        path = _unit_path(eval_dir, query, engine, seed) if eval_dir else None
        if path and resume and path.exists():
            if progress:
                progress(f"skip (already done): {engine}:{query[:34]}")
            return _load_unit(path)
        async with sem:
            for attempt in (1, 2):
                try:
                    rm, mms = await _run_once(query, engine)
                    if path:
                        _save_unit(path, rm, mms)  # durable the moment the screen finishes
                    if progress:
                        to = f", {rm.n_timeouts} timeout(s)" if rm.n_timeouts else ""
                        progress(f"{engine}:{query[:40]} — {rm.n_memos} memo(s){to}")
                    return rm, mms
                except Exception as e:  # noqa: BLE001 — eval resilience, never crash the matrix
                    if attempt == 2:
                        if progress:
                            progress(f"DROPPED {engine}:{query[:34]} after retry — {e}")
                        return None
        return None

    units = [
        (query, engine, seed)
        for query in queries
        for engine in engines
        for seed in range(seeds)
    ]
    raw = await asyncio.gather(*[_unit(q, e, s) for q, e, s in units])
    results = [r for r in raw if r is not None]
    run_metrics = [rm for rm, _ in results]
    memo_metrics = [m for _, mms in results for m in mms]
    return _summarize(engines, memo_metrics, run_metrics)


def _summarize(engines, memo_metrics, run_metrics) -> Comparison:
    summaries = []
    for engine in engines:
        mm = [m for m in memo_metrics if m.engine == engine]
        rm = [r for r in run_metrics if r.engine == engine]
        n_memos = sum(r.n_memos for r in rm)
        s = EngineSummary(
            engine=engine,
            n_runs=len(rm),
            n_memos=n_memos,
            n_failures=sum(r.n_failures for r in rm),
            n_timeouts=sum(r.n_timeouts for r in rm),
        )
        if mm:
            s.citation_accuracy = round(mean(m.citation_accuracy for m in mm), 4)
            s.grounding = round(mean(m.grounding for m in mm), 2)
            s.argument_balance = round(mean(m.argument_balance for m in mm), 2)
            s.specificity = round(mean(m.specificity for m in mm), 2)
            s.readability = round(mean(m.readability for m in mm), 2)
        if rm:
            total_cost = sum(r.cost_usd for r in rm)
            total_tokens = sum(r.total_tokens for r in rm)
            s.cost_per_memo = round(total_cost / n_memos, 4) if n_memos else 0.0
            s.tokens_per_memo = round(total_tokens / n_memos, 1) if n_memos else 0.0
            s.wall_clock_s = round(mean(r.wall_clock_s for r in rm), 2)
        summaries.append(s)
    return Comparison(engines=summaries, memo_metrics=memo_metrics, run_metrics=run_metrics)


def render_markdown(cmp: Comparison) -> str:
    lines = [
        "# Baseline vs Pipeline — Comparison",
        "",
        "| Metric | " + " | ".join(e.engine for e in cmp.engines) + " |",
        "| --- | " + " | ".join("---" for _ in cmp.engines) + " |",
    ]

    def row(label, fn):
        return "| " + label + " | " + " | ".join(fn(e) for e in cmp.engines) + " |"

    lines.append(row("runs", lambda e: str(e.n_runs)))
    lines.append(row("memos produced", lambda e: str(e.n_memos)))
    lines.append(row("failures", lambda e: str(e.n_failures)))
    lines.append(row("of which timeouts", lambda e: str(e.n_timeouts)))
    lines.append(row("citation accuracy", lambda e: f"{e.citation_accuracy:.1%}"))
    lines.append(row("grounding (1-5)", lambda e: f"{e.grounding:.2f}"))
    lines.append(row("argument balance (1-5)", lambda e: f"{e.argument_balance:.2f}"))
    lines.append(row("specificity (1-5)", lambda e: f"{e.specificity:.2f}"))
    lines.append(row("readability (1-5)", lambda e: f"{e.readability:.2f}"))
    lines.append(row("cost / memo ($)", lambda e: f"{e.cost_per_memo:.4f}"))
    lines.append(row("tokens / memo", lambda e: f"{e.tokens_per_memo:,.0f}"))
    lines.append(row("wall-clock / run (s)", lambda e: f"{e.wall_clock_s:.1f}"))
    lines.append("")
    return "\n".join(lines)
