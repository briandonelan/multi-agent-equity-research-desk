"""Deterministic checks over screener output.

Each check is a pure function from one or more :class:`ScreenResult` artifacts (plus the suite
query and, where relevant, the dimension map) to a small dataclass of numbers. No LLM judge is
involved: the screener runs the filters mechanically, so a wrong shortlist can only come from a
wrong intent->filters translation, and that is exactly what these checks measure.
"""

from __future__ import annotations

from desk.evals.screener.checks.coverage import CoverageScore, score_coverage
from desk.evals.screener.checks.golden import GoldenScore, score_golden
from desk.evals.screener.checks.paraphrase import ParaphraseScore, score_paraphrase
from desk.evals.screener.checks.perturb import PerturbScore, score_perturb
from desk.evals.screener.checks.relaxation import RelaxationScore, score_relaxation

__all__ = [
    "CoverageScore",
    "GoldenScore",
    "ParaphraseScore",
    "PerturbScore",
    "RelaxationScore",
    "score_coverage",
    "score_golden",
    "score_paraphrase",
    "score_perturb",
    "score_relaxation",
]
