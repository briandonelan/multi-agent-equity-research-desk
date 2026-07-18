"""Relaxation check: when nothing matches, does the screener loosen the right filter and say so?

Starve queries are written so fewer than two rows match the literal request. The screener is
supposed to relax the least-central filter once and disclose it twice — as a structured
``relaxations`` entry and in one sentence of ``interpreted_intent``. This check reports three
booleans; it is informational (reported, never a hard failure), because "which filter is least
central" is a judgement call.
"""

from __future__ import annotations

from dataclasses import dataclass

from desk.contracts.v1 import ScreenResult
from desk.evals.screener.suite import SuiteQuery

_RELAX_WORDS = ("relax", "loosen", "loosened", "widen", "widened", "broaden", "eased", "ease")


@dataclass
class RelaxationScore:
    query_id: str
    expected_field: str | None
    correct_field_relaxed: bool
    structured_disclosure: bool  # at least one relaxations[] entry
    prose_mentioned: bool  # interpreted_intent admits a relaxation

    @property
    def fully_disclosed(self) -> bool:
        return self.correct_field_relaxed and self.structured_disclosure and self.prose_mentioned


def score_relaxation(result: ScreenResult, q: SuiteQuery) -> RelaxationScore:
    relaxed_fields = {r.field for r in result.relaxations}
    structured = len(result.relaxations) > 0
    correct = q.expected_relaxation_field in relaxed_fields if q.expected_relaxation_field else False

    intent = (result.interpreted_intent or "").lower()
    prose = any(w in intent for w in _RELAX_WORDS)

    return RelaxationScore(
        query_id=q.id,
        expected_field=q.expected_relaxation_field,
        correct_field_relaxed=correct,
        structured_disclosure=structured,
        prose_mentioned=prose,
    )
