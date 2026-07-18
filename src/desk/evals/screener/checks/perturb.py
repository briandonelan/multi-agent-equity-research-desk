"""Perturbation check: does a one-dimension change in the request move the right filter?

A perturbation edits exactly one dimension of the base request (e.g. "improving" -> "deteriorating"
margins). A good translation moves the filter on the affected field and leaves the others alone. We
report two rates:

- ``targeted_move_rate`` — of the fields the perturbation should touch, how many actually changed.
- ``off_target_change_rate`` — of the other fields the base run filtered on, how many drifted.
"""

from __future__ import annotations

from dataclasses import dataclass

from desk.contracts.v1 import ScreenResult
from desk.evals.screener.checks._filters import changed_fields, filters_by_field
from desk.evals.screener.suite import Perturbation, SuiteQuery


@dataclass
class PerturbScore:
    query_id: str
    perturbation_text: str
    targeted_move_rate: float
    off_target_change_rate: float
    expected_moved: list[str]
    actually_moved: list[str]  # of the expected-moved fields, which changed
    off_target: list[str]  # fields that changed but should not have


def score_perturb(
    base: ScreenResult,
    perturbed: ScreenResult,
    pert: Perturbation,
    q: SuiteQuery,
) -> PerturbScore:
    expected_moved = {f for f in pert.expect_moved_fields}
    changed = changed_fields(base.filters_applied, perturbed.filters_applied)

    actually_moved = expected_moved & changed
    targeted_move_rate = len(actually_moved) / len(expected_moved) if expected_moved else 1.0

    # Off-target: any field that changed but was not supposed to. The rate is relative to every
    # non-target field either run filtered on, so "changed nothing extra" scores 0.0.
    base_fields = set(filters_by_field(base.filters_applied))
    perturbed_fields = set(filters_by_field(perturbed.filters_applied))
    off_target_universe = (base_fields | perturbed_fields) - expected_moved
    off_target = changed - expected_moved
    off_target_change_rate = (
        len(off_target) / len(off_target_universe) if off_target_universe else 0.0
    )

    return PerturbScore(
        query_id=q.id,
        perturbation_text=pert.text,
        targeted_move_rate=targeted_move_rate,
        off_target_change_rate=off_target_change_rate,
        expected_moved=sorted(expected_moved),
        actually_moved=sorted(actually_moved),
        off_target=sorted(off_target),
    )
