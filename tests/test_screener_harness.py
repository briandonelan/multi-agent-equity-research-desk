"""M7.3: the deterministic checks, the runner that drives the real screener against the fixture,
and the report. Every test is offline — the agent is a CallbackAgentRunner, and one test asserts
the whole path makes zero HTTP calls."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from desk.agents.base import CallbackAgentRunner
from desk.contracts.v1 import Candidate, Relaxation, ScreenFilter, ScreenResult
from desk.evals.screener import harness, report, runner
from desk.evals.screener.checks import (
    score_coverage,
    score_golden,
    score_paraphrase,
    score_perturb,
    score_relaxation,
)
from desk.evals.screener.scoring import evaluate_query
from desk.evals.screener.suite import Perturbation, SuiteQuery, load_dimension_map, load_suite

DIM_MAP = load_dimension_map()


def _sr(*, intent="i", filters=None, tickers=(), relaxations=None) -> ScreenResult:
    return ScreenResult(
        run_id="t",
        interpreted_intent=intent,
        filters_applied=[ScreenFilter(**f) for f in (filters or [])],
        relaxations=relaxations or [],
        candidates=[
            Candidate(ticker=t, cik="1", company_name=t, rationale="x") for t in tickers
        ]
        or [Candidate(ticker="TEC3", cik="1", company_name="x", rationale="x")],
    )


def _q(**kw) -> SuiteQuery:
    base = dict(id="qX", text="t", dimensions={}, golden_filters=[], expected_tickers=[])
    base.update(kw)
    return SuiteQuery.model_validate(base)


# --- golden ----------------------------------------------------------------------------------


def test_golden_perfect_recall_and_precision():
    q = _q(expected_tickers=["TEC3", "TEC6"])
    s = score_golden(_sr(tickers=["TEC3", "TEC6"]), q)
    assert s.recall == 1.0 and s.precision == 1.0
    assert s.missing == [] and s.unexpected == []


def test_golden_can_fail_on_miss_and_extra():
    q = _q(expected_tickers=["TEC3", "TEC6"], acceptable_extra=["TEC9"])
    s = score_golden(_sr(tickers=["TEC3", "TEC9", "IND1"]), q)
    assert s.recall == 0.5  # got TEC3 of {TEC3, TEC6}
    assert s.missing == ["TEC6"]
    assert s.unexpected == ["IND1"]  # TEC9 is acceptable_extra, not penalized
    assert s.precision == pytest.approx(2 / 3)  # 1 wrong of 3 picks


# --- coverage --------------------------------------------------------------------------------


def test_coverage_full_and_partial():
    q = _q(dimensions={"sector": "technology", "size": "mid-cap"})
    full = score_coverage(
        _sr(filters=[{"field": "sector", "op": "==", "value": "Technology"},
                     {"field": "market_cap", "op": "<", "value": 1e10}]),
        q, DIM_MAP,
    )
    assert full.coverage_rate == 1.0 and full.uncovered == []

    partial = score_coverage(
        _sr(filters=[{"field": "sector", "op": "==", "value": "Technology"}]), q, DIM_MAP
    )
    assert partial.coverage_rate == 0.5 and partial.uncovered == ["size"]


# --- paraphrase ------------------------------------------------------------------------------


def test_paraphrase_identical_sets_score_one():
    q = _q()
    same = [_sr(tickers=["TEC3", "TEC6"]) for _ in range(6)]
    assert score_paraphrase(same, q).mean_jaccard == 1.0


def test_paraphrase_can_fail_when_sets_diverge():
    q = _q()
    results = [_sr(tickers=["TEC3", "TEC6"]), _sr(tickers=["IND1", "IND2"])]
    assert score_paraphrase(results, q).mean_jaccard == 0.0


# --- perturbation ----------------------------------------------------------------------------


def test_perturb_moves_target_only():
    q = _q()
    pert = Perturbation(change={"size": "large-cap"}, text="large tech",
                        expect_moved_fields=["market_cap"])
    base = _sr(filters=[{"field": "sector", "op": "==", "value": "Technology"},
                        {"field": "market_cap", "op": "<", "value": 1e10}])
    good = _sr(filters=[{"field": "sector", "op": "==", "value": "Technology"},
                        {"field": "market_cap", "op": ">=", "value": 1e10}])
    s = score_perturb(base, good, pert, q)
    assert s.targeted_move_rate == 1.0
    assert s.off_target_change_rate == 0.0


def test_perturb_can_fail_missing_move_and_off_target_drift():
    q = _q()
    pert = Perturbation(change={"size": "large-cap"}, text="large tech",
                        expect_moved_fields=["market_cap"])
    base = _sr(filters=[{"field": "sector", "op": "==", "value": "Technology"},
                        {"field": "market_cap", "op": "<", "value": 1e10}])
    # market_cap unchanged (target missed) but sector drifted (off-target).
    bad = _sr(filters=[{"field": "sector", "op": "==", "value": "Healthcare"},
                       {"field": "market_cap", "op": "<", "value": 1e10}])
    s = score_perturb(base, bad, pert, q)
    assert s.targeted_move_rate == 0.0
    assert s.off_target_change_rate == 1.0  # sector was the only non-target field, and it moved


# --- relaxation ------------------------------------------------------------------------------


def _relax(field, lo_op, lo, hi_op, hi):
    return Relaxation(
        field=field,
        original=ScreenFilter(field=field, op=lo_op, value=lo),
        relaxed_to=ScreenFilter(field=field, op=hi_op, value=hi),
        reason="widened",
    )


def test_relaxation_fully_disclosed():
    q = _q(starve=True, expected_relaxation_field="market_cap")
    s = score_relaxation(
        _sr(intent="I relaxed the market cap floor to find matches.",
            relaxations=[_relax("market_cap", ">", 5e10, ">", 1e10)]),
        q,
    )
    assert s.correct_field_relaxed and s.structured_disclosure and s.prose_mentioned
    assert s.fully_disclosed


def test_relaxation_can_fail_wrong_field_and_silent():
    q = _q(starve=True, expected_relaxation_field="market_cap")
    s = score_relaxation(
        _sr(intent="Mega-cap cheap industrials.",  # no relaxation admitted in prose
            relaxations=[_relax("trailing_pe", "<", 8, "<", 15)]),  # wrong field
        q,
    )
    assert not s.correct_field_relaxed
    assert s.structured_disclosure  # it did relax *something*
    assert not s.prose_mentioned
    assert not s.fully_disclosed


# --- runner + scoring integration (real run_screener, fake agent) ----------------------------


def _canned_responder(mapping: dict[str, dict]):
    """Return the canned JSON whose key is a substring of the request prompt."""

    def responder(spec):
        for needle, payload in mapping.items():
            if needle in spec.prompt:
                return payload
        raise AssertionError(f"no canned response for prompt: {spec.prompt!r}")

    return responder


def _golden_payload(intent, filters, tickers, relaxations=None):
    out = {
        "interpreted_intent": intent,
        "filters_applied": filters,
        "candidates": [
            {"ticker": t, "cik": "1", "company_name": t, "rationale": "grounded"} for t in tickers
        ],
    }
    if relaxations is not None:
        out["relaxations"] = relaxations
    return out


@pytest.mark.asyncio
async def test_observe_and_evaluate_ordinary_query():
    q = next(x for x in load_suite() if x.id == "q01_smid_tech")
    tech = _golden_payload(
        "small and mid-cap technology",
        [{"field": "sector", "op": "==", "value": "Technology"},
         {"field": "market_cap", "op": "<", "value": 10000000000}],
        ["TEC3", "TEC6"],
    )
    large = _golden_payload(
        "large-cap technology",
        [{"field": "sector", "op": "==", "value": "Technology"},
         {"field": "market_cap", "op": ">=", "value": 10000000000}],
        ["TEC1", "TEC4"],
    )
    # The perturbation text mentions large-cap; every other phrasing gets the base result.
    responder = _canned_responder({"large-cap technology companies": large, "": tech})
    runner_obj = CallbackAgentRunner(responder)

    obs = await runner.observe_query(q, runner=runner_obj, run_id="EVAL1", repeats=1)
    ev = evaluate_query(q, obs, DIM_MAP)

    assert ev.golden_recall == 1.0 and ev.golden_precision == 1.0
    assert ev.coverage_rate == 1.0
    assert ev.paraphrase_jaccard == 1.0  # base + 5 paraphrases all identical
    assert len(ev.perturbations) == 1
    assert ev.perturbations[0].targeted_move_rate == 1.0
    assert ev.perturbations[0].off_target_change_rate == 0.0


@pytest.mark.asyncio
async def test_observe_and_evaluate_starve_query():
    q = next(x for x in load_suite() if x.id == "q10_starve_megacap_cheap_industrials")
    relaxed = _golden_payload(
        "Mega-cap cheap industrials; I relaxed the market-cap floor to surface matches.",
        [{"field": "sector", "op": "==", "value": "Industrials"},
         {"field": "market_cap", "op": ">", "value": 10000000000},
         {"field": "trailing_pe", "op": "<", "value": 8}],
        ["IND1"],
        relaxations=[{
            "field": "market_cap",
            "original": {"field": "market_cap", "op": ">", "value": 50000000000},
            "relaxed_to": {"field": "market_cap", "op": ">", "value": 10000000000},
            "reason": "No mega-cap matched; widened the size floor.",
        }],
    )
    runner_obj = CallbackAgentRunner(_canned_responder({"": relaxed}))

    obs = await runner.observe_query(q, runner=runner_obj, run_id="EVAL2", repeats=1)
    ev = evaluate_query(q, obs, DIM_MAP)

    assert ev.golden_recall is None  # starve queries are judged on relaxation, not golden
    assert ev.relaxation is not None
    assert ev.relaxation.correct_field_rate == 1.0
    assert ev.relaxation.fully_disclosed_rate == 1.0


@pytest.mark.asyncio
async def test_harness_makes_no_network_calls(respx_mock):
    """The screener eval must run entirely against the fixture — any HTTP is a leak."""
    catch_all = respx_mock.route().mock(side_effect=AssertionError("network call in eval!"))
    q = next(x for x in load_suite() if x.id == "q09_defensive_healthcare_steady")
    payload = _golden_payload(
        "defensive healthcare",
        [{"field": "sector", "op": "==", "value": "Healthcare"},
         {"field": "revenue_growth_yoy", "op": "between", "value": [0.0, 0.05]}],
        ["HLT1", "HLT5"],
    )
    runner_obj = CallbackAgentRunner(_canned_responder({"": payload}))
    evals = await harness.run_matrix(
        [None], [q], runner=runner_obj, run_id="EVAL3", repeats=1
    )
    assert evals[0].queries[0].golden_recall is not None
    assert not catch_all.called


# --- report ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_renders_and_roundtrips():
    q = next(x for x in load_suite() if x.id == "q01_smid_tech")
    tech = _golden_payload(
        "tech",
        [{"field": "sector", "op": "==", "value": "Technology"},
         {"field": "market_cap", "op": "<", "value": 10000000000}],
        ["TEC3", "TEC6"],
    )
    runner_obj = CallbackAgentRunner(_canned_responder({"": tech}))
    evals = await harness.run_matrix([None], [q], runner=runner_obj, run_id="EVAL4", repeats=1)

    md = report.render_report(evals, eval_run_id="EVAL4", repeats=1)
    assert "## Summary" in md and "<!-- ANALYSIS -->" in md and "q01_smid_tech" in md

    # JSON round-trip reproduces the same headline numbers (so --report can re-render offline).
    reloaded = report.from_json(report.to_json(evals))
    assert reloaded[0].mean_recall == evals[0].mean_recall
    assert reloaded[0].queries[0].coverage_rate == evals[0].queries[0].coverage_rate


def test_httpx_import_available():
    # Guard: the no-network test depends on respx intercepting httpx, which the data layer uses.
    assert hasattr(httpx, "Client")


# --- concurrency (M7 follow-up: bounded parallel invocation) ---------------------------------


class _CountingRunner:
    """An async fake that sleeps (yielding the loop) so gathered calls truly overlap, and records
    the peak number of simultaneous in-flight calls."""

    def __init__(self, payload: dict, delay: float = 0.02):
        self.payload = payload
        self.delay = delay
        self.in_flight = 0
        self.peak = 0
        self.total = 0

    async def run(self, spec, run_id):  # noqa: ANN001
        import json as _json

        from desk.agents.base import AgentResult

        self.in_flight += 1
        self.total += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        return AgentResult(text=_json.dumps(self.payload), model=spec.model)


def _tech_payload():
    return {
        "interpreted_intent": "tech",
        "filters_applied": [
            {"field": "sector", "op": "==", "value": "Technology"},
            {"field": "market_cap", "op": "<", "value": 10000000000},
        ],
        "candidates": [
            {"ticker": t, "cik": "1", "company_name": t, "rationale": "x"} for t in ("TEC3", "TEC6")
        ],
    }


@pytest.mark.asyncio
async def test_semaphore_bounds_in_flight_calls():
    q = next(x for x in load_suite() if x.id == "q01_smid_tech")  # base + 5 paraphrases + 1 pert
    r = _CountingRunner(_tech_payload())
    sem = asyncio.Semaphore(3)
    await runner.observe_query(q, runner=r, run_id="C1", repeats=1, semaphore=sem)
    assert r.total == 7  # base + 5 paraphrases + 1 perturbation
    assert r.peak > 1  # calls genuinely overlapped...
    assert r.peak <= 3  # ...but never exceeded the semaphore bound


@pytest.mark.asyncio
async def test_concurrency_one_is_serial():
    q = next(x for x in load_suite() if x.id == "q01_smid_tech")
    r = _CountingRunner(_tech_payload())
    await runner.observe_query(q, runner=r, run_id="C2", repeats=1, semaphore=asyncio.Semaphore(1))
    assert r.peak == 1  # concurrency=1 degrades to sequential


@pytest.mark.asyncio
async def test_parallel_and_serial_give_identical_scores():
    """Concurrency is a throughput knob, not a correctness one: results must match exactly."""
    q = next(x for x in load_suite() if x.id == "q01_smid_tech")
    responder = _canned_responder({"": _tech_payload()})

    serial = await harness.run_matrix(
        [None], [q], runner=CallbackAgentRunner(responder), run_id="S", repeats=2, concurrency=1
    )
    parallel = await harness.run_matrix(
        [None], [q], runner=CallbackAgentRunner(responder), run_id="P", repeats=2, concurrency=8
    )
    a, b = serial[0].queries[0], parallel[0].queries[0]
    assert (a.golden_recall, a.golden_precision, a.coverage_rate, a.paraphrase_jaccard) == (
        b.golden_recall, b.golden_precision, b.coverage_rate, b.paraphrase_jaccard,
    )
