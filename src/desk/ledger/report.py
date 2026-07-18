"""Cost roll-ups over the ledger: per-run/per-stage breakdowns and cross-run trends."""

from __future__ import annotations

from dataclasses import dataclass, field

from desk.contracts.v1 import RunCostSummary
from desk.ledger import db


@dataclass
class StageRollup:
    stage: str
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    computed_cost_usd: float = 0.0
    reported_cost_usd: float = 0.0
    degradations: list[str] = field(default_factory=list)


@dataclass
class RunRollup:
    run_id: str
    stages: list[StageRollup] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    n_llm_calls: int = 0
    n_tool_calls: int = 0
    n_truncated_tool_calls: int = 0
    cache_savings_usd: float = 0.0


def run_rollup(run_id: str) -> RunRollup:
    """Aggregate all ledger rows for one run into a per-stage + total roll-up."""
    roll = RunRollup(run_id=run_id)
    stage_map: dict[str, StageRollup] = {}
    with db.connect() as conn:
        for r in conn.execute("SELECT * FROM llm_calls WHERE run_id=? ORDER BY id", (run_id,)):
            s = stage_map.setdefault(r["stage"], StageRollup(stage=r["stage"]))
            s.n_calls += 1
            s.input_tokens += r["input_tokens"] or 0
            s.output_tokens += r["output_tokens"] or 0
            s.cache_read_tokens += r["cache_read_tokens"] or 0
            s.cache_write_tokens += r["cache_write_tokens"] or 0
            s.computed_cost_usd += r["computed_cost_usd"] or 0.0
            s.reported_cost_usd += r["reported_cost_usd"] or 0.0
            import json

            for d in json.loads(r["degradations"] or "[]"):
                s.degradations.append(d)
            roll.total_cost_usd += r["computed_cost_usd"] or 0.0
            roll.total_input_tokens += r["input_tokens"] or 0
            roll.total_output_tokens += r["output_tokens"] or 0
            roll.total_cache_read_tokens += r["cache_read_tokens"] or 0
            roll.n_llm_calls += 1
            # Cache savings: what those cached reads would have cost at full input price.
            roll.cache_savings_usd += _cache_savings(r["model"], r["cache_read_tokens"] or 0)

        for r in conn.execute("SELECT * FROM tool_calls WHERE run_id=?", (run_id,)):
            roll.n_tool_calls += 1
            if r["truncated"]:
                roll.n_truncated_tool_calls += 1

    roll.stages = [stage_map[k] for k in sorted(stage_map)]
    roll.total_cost_usd = round(roll.total_cost_usd, 6)
    roll.cache_savings_usd = round(roll.cache_savings_usd, 6)
    return roll


def _cache_savings(model: str, cache_read_tokens: int) -> float:
    """Difference between full input price and cache-read price for cached tokens."""
    if not cache_read_tokens:
        return 0.0
    full = db.compute_cost(model, input_tokens=cache_read_tokens)
    cached = db.compute_cost(model, cache_read_tokens=cache_read_tokens)
    return max(0.0, full - cached)


def build_cost_summary(run_id: str) -> RunCostSummary:
    """The RunCostSummary embedded in a memo footer."""
    roll = run_rollup(run_id)
    return RunCostSummary(
        run_id=run_id,
        total_cost_usd=roll.total_cost_usd,
        total_input_tokens=roll.total_input_tokens,
        total_output_tokens=roll.total_output_tokens,
        total_cache_read_tokens=roll.total_cache_read_tokens,
        n_calls=roll.n_llm_calls,
        by_stage_cost_usd={s.stage: round(s.computed_cost_usd, 6) for s in roll.stages},
    )


def run_total_tokens(run_id: str) -> int:
    """Input + output tokens for a run (excludes cache reads — cheap reused context, not new
    spend). Used by the budget controller to decide degradation."""
    with db.connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) AS t "
            "FROM llm_calls WHERE run_id=?",
            (run_id,),
        ).fetchone()
    return int(row["t"] or 0)


def all_runs() -> list[RunRollup]:
    """Per-run roll-ups across every run in the ledger, newest first."""
    with db.connect() as conn:
        run_ids = [
            r["run_id"]
            for r in conn.execute(
                "SELECT run_id, MAX(ts) AS last FROM llm_calls GROUP BY run_id ORDER BY last DESC"
            )
        ]
    return [run_rollup(rid) for rid in run_ids]
