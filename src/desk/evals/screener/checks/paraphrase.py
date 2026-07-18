"""Paraphrase check: does the screener give the same shortlist for reworded requests?

The base phrasing and its five paraphrases should all mean the same thing. We run the screener on
all six, then take the mean pairwise Jaccard overlap of the resulting ticker sets. 1.0 means every
phrasing produced an identical shortlist; a low number means the translation is phrasing-sensitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from desk.contracts.v1 import ScreenResult
from desk.evals.screener.suite import SuiteQuery


@dataclass
class ParaphraseScore:
    query_id: str
    mean_jaccard: float
    n_sets: int
    pairwise: list[float]
    sets: list[list[str]]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0  # two empty shortlists agree
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def score_paraphrase(results: list[ScreenResult], q: SuiteQuery) -> ParaphraseScore:
    """``results`` is the base run followed by one run per paraphrase, in suite order."""
    sets = [{c.ticker.upper() for c in r.candidates} for r in results]
    if len(sets) < 2:
        return ParaphraseScore(
            query_id=q.id,
            mean_jaccard=1.0,
            n_sets=len(sets),
            pairwise=[],
            sets=[sorted(s) for s in sets],
        )
    pairwise = [_jaccard(a, b) for a, b in combinations(sets, 2)]
    mean = sum(pairwise) / len(pairwise)
    return ParaphraseScore(
        query_id=q.id,
        mean_jaccard=mean,
        n_sets=len(sets),
        pairwise=pairwise,
        sets=[sorted(s) for s in sets],
    )
