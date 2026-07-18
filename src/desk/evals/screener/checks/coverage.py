"""Coverage check: did every annotated dimension of the request produce at least one filter?

Each query is labelled with the dimensions it exercises (size, valuation, sector, ...). The
dimension map says which metric fields legitimately express each dimension. A dimension is covered
if the screener applied at least one filter on one of those fields. A missed dimension means the
request was partly ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from desk.contracts.v1 import ScreenResult
from desk.evals.screener.suite import SuiteQuery


@dataclass
class CoverageScore:
    query_id: str
    covered: dict[str, bool] = field(default_factory=dict)
    coverage_rate: float = 1.0
    uncovered: list[str] = field(default_factory=list)


def score_coverage(
    result: ScreenResult, q: SuiteQuery, dimension_map: dict[str, list[str]]
) -> CoverageScore:
    filtered_fields = {f.field for f in result.filters_applied}
    covered: dict[str, bool] = {}
    for dim in q.dimensions:
        fields_for_dim = set(dimension_map.get(dim, []))
        covered[dim] = bool(fields_for_dim & filtered_fields)

    uncovered = sorted(d for d, ok in covered.items() if not ok)
    rate = (len(covered) - len(uncovered)) / len(covered) if covered else 1.0
    return CoverageScore(
        query_id=q.id,
        covered=covered,
        coverage_rate=rate,
        uncovered=uncovered,
    )
