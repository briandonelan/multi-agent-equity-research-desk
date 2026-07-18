"""Golden check: does the shortlist recover the labelled expected tickers?

Recall and precision are reported separately. ``acceptable_extra`` tickers are defensible
inclusions (the label author decided either answer is fine) and never count against precision.
"""

from __future__ import annotations

from dataclasses import dataclass

from desk.contracts.v1 import ScreenResult
from desk.evals.screener.suite import SuiteQuery


@dataclass
class GoldenScore:
    query_id: str
    recall: float
    precision: float
    got: list[str]
    expected: list[str]
    missing: list[str]  # expected but absent
    unexpected: list[str]  # present but not in expected or acceptable_extra


def score_golden(result: ScreenResult, q: SuiteQuery) -> GoldenScore:
    got = {c.ticker.upper() for c in result.candidates}
    expected = {t.upper() for t in q.expected_tickers}
    acceptable = {t.upper() for t in q.acceptable_extra}

    hit = got & expected
    missing = expected - got
    unexpected = got - expected - acceptable

    recall = len(hit) / len(expected) if expected else 1.0
    # Precision counts only the genuinely wrong picks; expected+acceptable are both "correct".
    precision = (len(got) - len(unexpected)) / len(got) if got else 1.0

    return GoldenScore(
        query_id=q.id,
        recall=recall,
        precision=precision,
        got=sorted(got),
        expected=sorted(expected),
        missing=sorted(missing),
        unexpected=sorted(unexpected),
    )
