"""Aggregate per-seed check outputs into per-query and per-model evaluations.

Repeats are averaged: each seed is scored independently and the headline numbers are the mean
across seeds, with representative detail (missing/unexpected tickers, uncovered dimensions) taken
from the first non-errored seed. Which checks apply depends on the query — golden on ordinary
queries, relaxation on starve queries, perturbation only where perturbations are defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from desk.evals.screener.checks import (
    score_coverage,
    score_golden,
    score_paraphrase,
    score_perturb,
    score_relaxation,
)
from desk.evals.screener.runner import SeedObservation
from desk.evals.screener.suite import SuiteQuery


@dataclass
class PerturbSummary:
    text: str
    targeted_move_rate: float
    off_target_change_rate: float


@dataclass
class RelaxationSummary:
    expected_field: str | None
    correct_field_rate: float
    structured_rate: float
    prose_rate: float
    fully_disclosed_rate: float


@dataclass
class QueryEvaluation:
    query_id: str
    starve: bool
    n_seeds: int
    n_errors: int
    golden_recall: float | None = None
    golden_precision: float | None = None
    golden_missing: list[str] = field(default_factory=list)
    golden_unexpected: list[str] = field(default_factory=list)
    coverage_rate: float | None = None
    coverage_uncovered: list[str] = field(default_factory=list)
    paraphrase_jaccard: float | None = None
    perturbations: list[PerturbSummary] = field(default_factory=list)
    relaxation: RelaxationSummary | None = None


def _mean_or_none(xs: list[float]) -> float | None:
    return mean(xs) if xs else None


def evaluate_query(
    q: SuiteQuery,
    observations: list[SeedObservation],
    dimension_map: dict[str, list[str]],
) -> QueryEvaluation:
    good = [o for o in observations if o.error is None and o.base is not None]
    n_errors = len(observations) - len(good)
    ev = QueryEvaluation(
        query_id=q.id,
        starve=q.starve,
        n_seeds=len(observations),
        n_errors=n_errors,
    )
    if not good:
        return ev

    # --- golden (ordinary queries only) / relaxation (starve queries only) ---
    if q.starve:
        rel = [score_relaxation(o.base, q) for o in good]
        ev.relaxation = RelaxationSummary(
            expected_field=q.expected_relaxation_field,
            correct_field_rate=mean(r.correct_field_relaxed for r in rel),
            structured_rate=mean(r.structured_disclosure for r in rel),
            prose_rate=mean(r.prose_mentioned for r in rel),
            fully_disclosed_rate=mean(r.fully_disclosed for r in rel),
        )
    else:
        golden = [score_golden(o.base, q) for o in good]
        ev.golden_recall = mean(g.recall for g in golden)
        ev.golden_precision = mean(g.precision for g in golden)
        ev.golden_missing = golden[0].missing
        ev.golden_unexpected = golden[0].unexpected

    # --- coverage (all queries) ---
    cov = [score_coverage(o.base, q, dimension_map) for o in good]
    ev.coverage_rate = mean(c.coverage_rate for c in cov)
    ev.coverage_uncovered = cov[0].uncovered

    # --- paraphrase (queries with paraphrases) ---
    para_scores: list[float] = []
    for o in good:
        if o.paraphrases and all(p is not None for p in o.paraphrases):
            ps = score_paraphrase([o.base, *o.paraphrases], q)
            para_scores.append(ps.mean_jaccard)
    ev.paraphrase_jaccard = _mean_or_none(para_scores)

    # --- perturbation (queries with perturbations), aligned by index ---
    for i, pert in enumerate(q.perturbations):
        moved: list[float] = []
        off: list[float] = []
        for o in good:
            if i < len(o.perturbations) and o.perturbations[i] is not None:
                ps = score_perturb(o.base, o.perturbations[i], pert, q)
                moved.append(ps.targeted_move_rate)
                off.append(ps.off_target_change_rate)
        if moved:
            ev.perturbations.append(
                PerturbSummary(
                    text=pert.text,
                    targeted_move_rate=mean(moved),
                    off_target_change_rate=mean(off),
                )
            )
    return ev


@dataclass
class ModelEvaluation:
    """One model's results across the whole suite."""

    model: str
    queries: list[QueryEvaluation] = field(default_factory=list)

    def _collect(self, attr: str) -> list[float]:
        return [v for q in self.queries if (v := getattr(q, attr)) is not None]

    @property
    def mean_recall(self) -> float | None:
        return _mean_or_none(self._collect("golden_recall"))

    @property
    def mean_precision(self) -> float | None:
        return _mean_or_none(self._collect("golden_precision"))

    @property
    def mean_coverage(self) -> float | None:
        return _mean_or_none(self._collect("coverage_rate"))

    @property
    def mean_paraphrase(self) -> float | None:
        return _mean_or_none(self._collect("paraphrase_jaccard"))

    @property
    def mean_targeted_move(self) -> float | None:
        rates = [p.targeted_move_rate for q in self.queries for p in q.perturbations]
        return _mean_or_none(rates)

    @property
    def mean_off_target(self) -> float | None:
        rates = [p.off_target_change_rate for q in self.queries for p in q.perturbations]
        return _mean_or_none(rates)

    @property
    def relaxation_disclosed_rate(self) -> float | None:
        rates = [q.relaxation.fully_disclosed_rate for q in self.queries if q.relaxation]
        return _mean_or_none(rates)

    @property
    def n_errors(self) -> int:
        return sum(q.n_errors for q in self.queries)
