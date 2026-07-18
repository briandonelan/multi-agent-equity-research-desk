"""LLM-as-judge memo quality scoring.

Two parts:
- **Programmatic** citation accuracy: each memo citation is spot-verified against the cached
  passage (resolve + fuzzy quote match) in code, NOT by the judge.
- **LLM judge** (Opus-tier): scores grounding, argument balance, specificity, and readability
  1-5 with a one-paragraph justification. The judge NEVER sees which engine produced the memo —
  provenance (run_id, produced_by, cost) is stripped before rendering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pydantic import Field

from desk.agents.base import AgentRunner, AgentStage, quote_matches, resolve_passage
from desk.contracts.v1 import Artifact, ResearchMemo
from desk.settings import load_yaml_config

JUDGE_SYSTEM_PROMPT = """\
You are an impartial research-quality judge. You score a single equity research memo on four
dimensions, each an integer 1-5. You do not know or care which system produced it.

Score against these anchors. A 5 requires the memo to clear the top bar with NO weakness on that
dimension; if you can name even one shortcoming, it is at most a 4. Reserve 1 for a genuine
failure. Levels 2 and 4 fall between the anchors below.

grounding — are claims tied to specific, resolvable evidence?
  1: claims are mostly bare assertions; citations missing, generic, or not supporting the claim.
  3: most claims are cited, but at least one load-bearing claim rests on a vague or loosely
     related quote.
  5: every claim is tied to a specific citation whose quoted text directly supports it; no
     unsupported assertions anywhere.

argument_balance — does the bear case materially engage the bull case?
  1: bear case is absent, a token disclaimer, or boilerplate ("risks include market conditions").
  3: bear case raises real risks, but they run parallel to the bull thesis rather than contesting
     its specific pillars; or one side is much thinner than the other.
  5: the bear case directly rebuts the specific pillars of the bull thesis with evidence of
     comparable weight; a reader feels genuine, unresolved tension.

specificity — concrete detail vs. generic language?
  1: mostly platitudes; few or no specific numbers, periods, segments, or named risks.
  3: a mix — some concrete figures alongside generic filler that could describe any company.
  5: concrete figures, named periods, and named risks/segments throughout; nothing could be
     copy-pasted onto a different company.

readability — is the rendered memo clear and well-organized?
  1: disorganized, repetitive, or hard to follow.
  3: understandable but uneven — some redundancy, awkward organization, or a buried main point.
  5: clear thesis, logical flow, quick to read in a single pass.

Before scoring, weigh the memo against each anchor; in `justification`, give one sentence per
dimension naming the specific evidence (or shortcoming) that fixed its score. Be discriminating:
identical scores across all four dimensions are rare and usually mean you did not look hard enough.

Output ONLY this JSON (no prose, no code fence):
{
  "grounding": int, "argument_balance": int, "specificity": int, "readability": int,
  "justification": str
}
Do not comment on investment merit — only memo quality.
"""


class JudgeReport(Artifact):
    grounding: int = Field(ge=1, le=5)
    argument_balance: int = Field(ge=1, le=5)
    specificity: int = Field(ge=1, le=5)
    readability: int = Field(ge=1, le=5)
    justification: str


@dataclass
class CitationAccuracy:
    n_citations: int
    n_verified: int

    @property
    def accuracy(self) -> float:
        return (self.n_verified / self.n_citations) if self.n_citations else 1.0


@dataclass
class MemoScore:
    ticker: str
    citation_accuracy: float
    n_citations: int
    n_verified: int
    grounding: int
    argument_balance: int
    specificity: int
    readability: int
    justification: str

    def as_dict(self) -> dict:
        return asdict(self)


def programmatic_citation_accuracy(memo: ResearchMemo) -> CitationAccuracy:
    """Fraction of memo citations whose quote resolves + matches its cached passage."""
    cites = []
    for claim in memo.bull_case:
        cites.extend(claim.citations)
    for ch in memo.bear_case:
        cites.extend(ch.citations)
    verified = 0
    for c in cites:
        passage = resolve_passage(c.source_ref)
        if passage is not None and quote_matches(c.quote, passage):
            verified += 1
    return CitationAccuracy(n_citations=len(cites), n_verified=verified)


def _memo_for_judge(memo: ResearchMemo) -> str:
    """Render the memo WITHOUT provenance so the judge can't infer the engine."""
    lines = [f"# {memo.company_name} ({memo.ticker})", "", "## Thesis", memo.thesis_summary, ""]
    lines.append("## Bull case")
    for i, claim in enumerate(memo.bull_case, 1):
        lines.append(f"{i}. [{claim.claim_type}] {claim.text}")
        for c in claim.citations:
            lines.append(f'   - "{c.quote}" ({c.source_ref})')
    lines.append("")
    lines.append("## Bear case")
    for i, ch in enumerate(memo.bear_case, 1):
        lines.append(f"{i}. [{ch.severity}] {ch.text}")
        for c in ch.citations:
            lines.append(f'   - "{c.quote}" ({c.source_ref})')
    lines.append("")
    lines.append("## Unresolved disagreements")
    for d in memo.unresolved_disagreements:
        lines.append(f"- {d}")
    lines.append("")
    lines.append(f"## Confidence: {memo.confidence}")
    lines.append(memo.confidence_rationale)
    return "\n".join(lines)


async def judge_memo(
    memo: ResearchMemo,
    *,
    run_id: str,
    runner: AgentRunner,
    model: str | None = None,
) -> MemoScore:
    cit = programmatic_citation_accuracy(memo)

    models = load_yaml_config("models").get("stages", {})
    stage = AgentStage(
        name="judge",
        model=model or models.get("judge", "claude-opus-4-8"),
        system_prompt=JUDGE_SYSTEM_PROMPT,
        runner=runner,
        run_id=run_id,
        allowed_tools=[],
        mcp_servers={},
        max_turns=1,
    )
    report = await stage.run(
        f"Score this memo:\n\n{_memo_for_judge(memo)}\n\nReturn the JudgeReport JSON.",
        contract_cls=JudgeReport,
    )
    assert isinstance(report, JudgeReport)

    return MemoScore(
        ticker=memo.ticker,
        citation_accuracy=round(cit.accuracy, 4),
        n_citations=cit.n_citations,
        n_verified=cit.n_verified,
        grounding=report.grounding,
        argument_balance=report.argument_balance,
        specificity=report.specificity,
        readability=report.readability,
        justification=report.justification,
    )
