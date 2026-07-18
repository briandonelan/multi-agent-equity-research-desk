"""Engine dispatch + memo persistence. The ``baseline`` engine runs a single agent; the
``pipeline`` engine runs the multi-agent DAG (screen -> reader/critic -> synth).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from desk.agents.base import AgentRunner, SdkAgentRunner
from desk.contracts.v1 import HandoffFailure, ResearchMemo
from desk.orchestrator.render import render_memo
from desk.orchestrator.run import RunContext, new_run_id


@dataclass
class RunResult:
    run_id: str
    engine: str
    memos: list[ResearchMemo] = field(default_factory=list)
    memo_paths: list[Path] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)


async def run_memo(
    query: str,
    *,
    engine: str = "pipeline",
    max_candidates: int = 4,
    handoff_mode: str | None = None,
    runner: AgentRunner | None = None,
) -> RunResult:
    """Run a memo request end-to-end and persist artifacts under ``runs/{run_id}/``."""
    run_id = new_run_id()
    ctx = RunContext(run_id, engine=engine, query=query)
    runner = runner or SdkAgentRunner()
    result = RunResult(run_id=run_id, engine=engine)

    if engine == "baseline":
        from desk.agents.baseline import run_baseline

        try:
            memos = await run_baseline(
                query, run_id=run_id, runner=runner, run_ctx=ctx, max_candidates=max_candidates
            )
        except HandoffFailure as exc:
            ctx.write_failure("baseline", exc.to_record())
            result.failures.append(exc.to_record())
            memos = []
        except Exception as exc:  # noqa: BLE001
            # A baseline that errors out (e.g. exhausts its turn budget) came back empty-handed;
            # that is a real, recordable outcome, not a crash to drop. Record 0 memos + the
            # failure so the coverage comparison counts it honestly.
            is_timeout = isinstance(exc, (asyncio.TimeoutError, TimeoutError))
            rec = {
                "stage": "baseline",
                "errors": [f"{type(exc).__name__}: {exc}"],
                "raw_output": "",
                "timeout": is_timeout,
            }
            ctx.write_failure("baseline", rec)
            result.failures.append(rec)
            memos = []
    elif engine == "pipeline":
        from desk.orchestrator.multi_agent import run_pipeline

        memos, failures = await run_pipeline(
            query,
            run_id=run_id,
            runner=runner,
            run_ctx=ctx,
            max_candidates=max_candidates,
            handoff_mode=handoff_mode,
        )
        result.failures.extend(failures)
    else:
        raise ValueError(f"Unknown engine: {engine!r}")

    for memo in memos:
        ctx.write_artifact(f"memo_{memo.ticker}", memo)
        markdown = render_memo(memo)
        path = ctx.write_memo(memo.ticker, markdown, memo)
        result.memos.append(memo)
        result.memo_paths.append(path)

    # Terminal marker: written last, so its presence distinguishes a finished run (even an
    # empty one) from a run killed mid-flight, and records the authoritative failure count.
    ctx.write_result(
        n_memos=len(result.memos),
        n_failures=len(result.failures),
        memo_tickers=[m.ticker for m in result.memos],
        failures=result.failures,
    )
    return result
