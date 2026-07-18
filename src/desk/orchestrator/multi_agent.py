"""The multi-agent DAG: screen -> (reader -> critic) per ticker -> synth -> memo.

Explicit async Python — no supervisor LLM. Per-ticker chains run concurrently (bounded by a
semaphore). Before each stage the orchestrator consults a :class:`BudgetController`: if the
run's running token total exceeds a stage's soft cap it engages the degradation ladder (reduce
truncation -> drop model tier -> limit filings); a hard cap flags ``budget_exhausted`` and stops
launching new candidate chains. Failure propagation:
- reader fails      -> the ticker is dropped, with a recorded failure note.
- critic fails      -> the memo is still produced with an empty bear_case and an explicit
                       unresolved_disagreements entry saying critique was unavailable.
- synth fails       -> that memo is dropped, with a recorded failure note.
Each stage is bounded by a timeout; timeouts are recorded like validation failures.
"""

from __future__ import annotations

import asyncio

from desk.agents.base import AgentRunner
from desk.agents.critic import run_critic
from desk.agents.reader import run_reader
from desk.agents.screener import run_screener
from desk.agents.synthesizer import run_synthesizer
from desk.contracts.v1 import Candidate, CritiqueReport, HandoffFailure, ResearchMemo, ThesisDraft
from desk.ledger import report
from desk.orchestrator.policies import BudgetController, StagePlan, plan_stage
from desk.orchestrator.run import RunContext
from desk.settings import load_yaml_config

BASE_MAX_SECTION_CHARS = 12_000


def _is_timeout(exc: Exception) -> bool:
    # asyncio.TimeoutError is an alias of TimeoutError on 3.11+, but check both for safety.
    return isinstance(exc, (asyncio.TimeoutError, TimeoutError))


def _fail_record(stage: str, exc: Exception) -> dict:
    if isinstance(exc, HandoffFailure):
        rec = exc.to_record()
        rec["stage"] = stage
        rec["timeout"] = False
        return rec
    # A timeout is an operational limit (the stage ran out of wall clock), NOT the validator
    # rejecting bad output. Flag it so the scoreboard can separate the two.
    return {
        "stage": stage,
        "errors": [f"{type(exc).__name__}: {exc}"],
        "raw_output": "",
        "timeout": _is_timeout(exc),
    }


async def _with_timeout(coro, seconds: float):
    return await asyncio.wait_for(coro, timeout=seconds)


class _Plans:
    """Builds per-stage plans from the budget controller + model/tier config."""

    def __init__(self, controller: BudgetController, models: dict, tier_order: list[str]):
        self.controller = controller
        self.models = models
        self.tier_order = tier_order
        self.factor = controller.truncation_factor

    def for_stage(self, stage: str, run_id: str) -> StagePlan:
        decision = self.controller.evaluate(stage, report.run_total_tokens(run_id))
        return plan_stage(
            decision=decision,
            model=self.models.get(stage, ""),
            base_max_section_chars=BASE_MAX_SECTION_CHARS,
            tier_order=self.tier_order,
            truncation_factor=self.factor,
        )


async def run_pipeline(
    query: str,
    *,
    run_id: str,
    runner: AgentRunner,
    run_ctx: RunContext,
    max_candidates: int = 4,
    handoff_mode: str | None = None,
    controller: BudgetController | None = None,
) -> tuple[list[ResearchMemo], list[dict]]:
    budgets = load_yaml_config("budgets")
    models_cfg = load_yaml_config("models")
    concurrency = int(budgets.get("concurrency", 2)) or 2
    timeout = float(budgets.get("defaults", {}).get("timeout_seconds", 180))
    handoff_mode = handoff_mode or budgets.get("handoff_mode", "contract")

    plans = _Plans(
        controller or BudgetController.from_config(budgets),
        models_cfg.get("stages", {}),
        models_cfg.get("tier_order", []),
    )

    failures: list[dict] = []

    # 1. Screen.
    sp = plans.for_stage("screener", run_id)
    try:
        screen = await _with_timeout(
            run_screener(
                query,
                run_id=run_id,
                runner=runner,
                max_candidates=max_candidates,
                run_ctx=run_ctx,
                model=sp.model or None,
                max_section_chars=sp.max_section_chars,
                degradations=sp.degradations,
            ),
            timeout,
        )
    except (HandoffFailure, TimeoutError) as exc:
        rec = _fail_record("screener", exc)
        run_ctx.write_failure("screener", rec)
        return [], [rec]
    run_ctx.write_artifact("screen_result", screen)

    candidates = screen.candidates[:max_candidates]
    sem = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()  # set when a hard cap is hit -> stop launching new chains

    async def process(cand: Candidate):
        if stop.is_set():
            return None, []
        async with sem:
            return await _process_candidate(
                cand, run_id, runner, run_ctx, handoff_mode, timeout, plans, stop
            )

    results = await asyncio.gather(*(process(c) for c in candidates), return_exceptions=True)

    memos: list[ResearchMemo] = []
    for res in results:
        if isinstance(res, BaseException):
            failures.append({"stage": "pipeline", "errors": [str(res)], "raw_output": ""})
            continue
        memo, fs = res
        failures.extend(fs)
        if memo is not None:
            memos.append(memo)
    return memos, failures


async def _process_candidate(
    cand: Candidate,
    run_id: str,
    runner: AgentRunner,
    run_ctx: RunContext,
    handoff_mode: str,
    timeout: float,
    plans: _Plans,
    stop: asyncio.Event,
) -> tuple[ResearchMemo | None, list[dict]]:
    fs: list[dict] = []
    exhausted = False

    # 2. Reader.
    rp = plans.for_stage("reader", run_id)
    exhausted = exhausted or rp.budget_exhausted
    try:
        thesis: ThesisDraft = await _with_timeout(
            run_reader(
                cand,
                run_id=run_id,
                runner=runner,
                run_ctx=run_ctx,
                model=rp.model or None,
                max_section_chars=rp.max_section_chars,
                max_filings=rp.max_filings,
                degradations=rp.degradations,
            ),
            timeout,
        )
    except (HandoffFailure, TimeoutError) as exc:
        rec = _fail_record(f"reader:{cand.ticker}", exc)
        run_ctx.write_failure(f"reader_{cand.ticker}", rec)
        return None, [rec]  # drop this ticker
    run_ctx.write_artifact(f"thesis_{cand.ticker}", thesis)

    # 3. Critic (adversarial, isolated). Failure -> memo without a bear case (surfaced below).
    cp = plans.for_stage("critic", run_id)
    exhausted = exhausted or cp.budget_exhausted
    critique: CritiqueReport | None = None
    try:
        critique = await _with_timeout(
            run_critic(
                thesis,
                run_id=run_id,
                runner=runner,
                handoff_mode=handoff_mode,
                run_ctx=run_ctx,
                model=cp.model or None,
                max_section_chars=cp.max_section_chars,
                degradations=cp.degradations,
            ),
            timeout,
        )
        run_ctx.write_artifact(f"critique_{cand.ticker}", critique)
    except (HandoffFailure, TimeoutError) as exc:
        rec = _fail_record(f"critic:{cand.ticker}", exc)
        run_ctx.write_failure(f"critic_{cand.ticker}", rec)
        fs.append(rec)
        critique = None

    # 4. Synthesize.
    syp = plans.for_stage("synthesizer", run_id)
    exhausted = exhausted or syp.budget_exhausted
    try:
        memo = await _with_timeout(
            run_synthesizer(
                cand,
                thesis,
                critique,
                run_id=run_id,
                runner=runner,
                run_ctx=run_ctx,
                model=syp.model or None,
                degradations=syp.degradations,
            ),
            timeout,
        )
    except (HandoffFailure, TimeoutError) as exc:
        rec = _fail_record(f"synthesizer:{cand.ticker}", exc)
        run_ctx.write_failure(f"synthesizer_{cand.ticker}", rec)
        fs.append(rec)
        return None, fs

    # A memo produced without a critique must say so loudly.
    if critique is None:
        note = "Adversarial critique was unavailable for this memo (critic stage failed)."
        if note not in memo.unresolved_disagreements:
            memo.unresolved_disagreements.append(note)

    # Hard cap: flag the memo and stop launching new candidate chains.
    if exhausted:
        stop.set()
        note = "Token hard cap reached during this run; results may be partial (budget_exhausted)."
        if note not in memo.unresolved_disagreements:
            memo.unresolved_disagreements.append(note)

    return memo, fs
