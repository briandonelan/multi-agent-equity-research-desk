"""Versioned handoff + memo contracts.

These Pydantic models are the ONLY things passed between agent stages. Every artifact carries
provenance (``schema_version``, ``run_id``, ``created_at``, ``produced_by``) and a
``token_cost`` stamp added by the runner. Semantic validation (citations resolve to cached
passages, quotes match, tickers in-universe) lives in ``agents/base.py`` — the models here
enforce only shape and length bounds.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"

Op = Literal["<", "<=", ">", ">=", "==", "between"]
ClaimType = Literal["fact", "interpretation", "projection"]
Severity = Literal["minor", "material", "fatal"]
Assessment = Literal["thesis_holds", "thesis_weakened", "thesis_rejected"]
Confidence = Literal["low", "medium", "high"]

DISCLAIMER = (
    "This memo is an engineering demonstration of multi-agent orchestration. Nothing in it is "
    "investment advice or a recommendation to transact in any security. It describes evidence "
    "and disagreement drawn from public filings; it is not a research recommendation."
)


def _now() -> datetime:
    return datetime.now(UTC)


# --- Cost stamps ----------------------------------------------------------------------------


class StageCost(BaseModel):
    """Per-call cost stamp attached to each artifact by the runner."""

    stage: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    computed_cost_usd: float = 0.0
    reported_cost_usd: float | None = None
    latency_ms: int = 0
    n_turns: int = 0
    degradations: list[str] = Field(default_factory=list)
    budget_exhausted: bool = False


class RunCostSummary(BaseModel):
    """Roll-up embedded in a memo: 'this memo cost $X across N calls'."""

    run_id: str
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    n_calls: int = 0
    by_stage_cost_usd: dict[str, float] = Field(default_factory=dict)


# --- Artifact base --------------------------------------------------------------------------


class Artifact(BaseModel):
    """Common provenance for every cross-stage artifact."""

    schema_version: str = SCHEMA_VERSION
    run_id: str
    created_at: datetime = Field(default_factory=_now)
    produced_by: str = ""  # "<agent-name> (<model>)"
    token_cost: StageCost | None = None


# --- Screener -------------------------------------------------------------------------------


class ScreenRequest(Artifact):
    """CLI -> screener."""

    query: str
    max_candidates: int = 4


class ScreenFilter(BaseModel):
    field: str
    op: Op
    # str supports categorical filters like sector == "Industrials"; tuple is for "between".
    value: float | str | tuple[float, float]


class Candidate(BaseModel):
    ticker: str
    cik: str
    company_name: str
    rationale: str = Field(max_length=600)  # <= ~80 words
    # Values are usually numeric, but a snapshot legitimately carries categorical fields like
    # sector; allow str so a whole screen doesn't fail validation over one label. (Relaxation is
    # backward-compatible: float-only snapshots from older runs still validate.)
    metrics_snapshot: dict[str, float | str | None] = Field(default_factory=dict)


class Relaxation(BaseModel):
    """A filter the screener loosened because the original was too strict (schema v1.1)."""

    field: str
    original: ScreenFilter
    relaxed_to: ScreenFilter
    reason: str = Field(max_length=200)  # <= ~30 words


class ScreenResult(Artifact):
    """Screener -> orchestrator.

    schema_version "1.1" adds the structured ``relaxations`` list. The change is additive and
    backward-compatible: a v1 artifact (no ``relaxations`` key) loads with an empty list.
    """

    schema_version: str = "1.1"
    interpreted_intent: str
    filters_applied: list[ScreenFilter] = Field(default_factory=list)
    relaxations: list[Relaxation] = Field(default_factory=list)  # NEW in v1.1
    candidates: list[Candidate] = Field(min_length=1, max_length=5)


# --- Reader ---------------------------------------------------------------------------------


class Citation(BaseModel):
    source_ref: str  # accession#item¶idx — must resolve to a cached passage (validated)
    quote: str = Field(max_length=320)  # <= ~40 words, verbatim from source


class Claim(BaseModel):
    text: str
    citations: list[Citation] = Field(min_length=1)
    claim_type: ClaimType


class ThesisDraft(Artifact):
    """Reader -> critic (per ticker)."""

    ticker: str
    thesis_summary: str = Field(max_length=900)  # <= ~120 words
    claims: list[Claim] = Field(min_length=4, max_length=8)

    def for_critic(self) -> ThesisDraftForCritic:
        """Deterministic isolation view: drop the summary and all reader reasoning;
        pass only claims (text + citations). This is the sycophancy control."""
        return ThesisDraftForCritic(
            run_id=self.run_id,
            ticker=self.ticker,
            claims=[
                Claim(text=c.text, citations=c.citations, claim_type=c.claim_type)
                for c in self.claims
            ],
        )


class ThesisDraftForCritic(BaseModel):
    """What the critic sees under isolation: no summary, no confidence, no reasoning."""

    run_id: str
    ticker: str
    claims: list[Claim]


# --- Critic ---------------------------------------------------------------------------------


class Challenge(BaseModel):
    target_claim_idx: int | None = None  # None = challenges the thesis as a whole
    text: str
    citations: list[Citation] = Field(default_factory=list)
    severity: Severity


class CritiqueReport(Artifact):
    """Critic -> synthesizer."""

    ticker: str
    challenges: list[Challenge] = Field(min_length=2, max_length=6)
    overall_assessment: Assessment
    what_would_change_my_mind: str


# --- Synthesizer / final memo ---------------------------------------------------------------


class ResearchMemo(Artifact):
    """Synthesizer -> user (also rendered to Markdown)."""

    ticker: str
    company_name: str
    as_of: date
    thesis_summary: str
    valuation_snapshot: dict[str, float | None] = Field(default_factory=dict)
    bull_case: list[Claim] = Field(default_factory=list)
    bear_case: list[Challenge] = Field(default_factory=list)
    unresolved_disagreements: list[str] = Field(default_factory=list)
    confidence: Confidence
    confidence_rationale: str
    cost: RunCostSummary
    disclaimer: str = DISCLAIMER


# --- Failure record -------------------------------------------------------------------------


class HandoffFailure(Exception):
    """Raised when a boundary fails validation twice (parse + one repair retry)."""

    def __init__(self, stage: str, errors: list[str], raw_output: str):
        self.stage = stage
        self.errors = errors
        self.raw_output = raw_output
        super().__init__(f"Handoff failed at stage {stage!r}: {errors}")

    def to_record(self) -> dict:
        return {
            "stage": self.stage,
            "errors": self.errors,
            "raw_output": self.raw_output[:4000],
        }
