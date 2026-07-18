"""Budget enforcement and the degradation ladder.

Before each stage the orchestrator asks a :class:`BudgetController` what to do, given the run's
running token total and the stage's soft/hard caps:

- **under soft cap** -> run normally.
- **soft cap exceeded** -> engage the next rung of the degradation ladder (cumulative across the
  run): (1) reduce tool truncation, (2) drop the stage to the next-cheaper model tier, (3) for the
  reader, limit to the 2 most recent filings. Every decision is logged to the ledger with a reason
  (via the stage's ``degradations`` stamp).
- **hard cap exceeded** -> the stage is flagged ``budget_exhausted`` and the run stops launching
  new candidate chains; the synthesizer surfaces this in the memo.

Token budgeting counts input + output tokens (the model's own work), excluding cache reads — a
cache hit is cheap reused context, not new spend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RUNG_REDUCE_TRUNCATION = "reduce_tool_truncation"
RUNG_DROP_TIER = "drop_model_tier"
RUNG_LIMIT_FILINGS = "limit_recent_filings"


@dataclass
class BudgetDecision:
    degradations: list[str] = field(default_factory=list)
    budget_exhausted: bool = False

    def reasons(self, stage: str, spend: int, soft: int | None, hard: int | None) -> list[str]:
        out = []
        if self.budget_exhausted:
            out.append(f"{stage}: hard cap {hard} exceeded at {spend} tokens (budget_exhausted)")
        for rung in self.degradations:
            out.append(f"{stage}: soft cap {soft} exceeded at {spend} tokens -> {rung}")
        return out


class BudgetController:
    """Stateful ladder controller for one run. Rungs engage cumulatively as spend crosses caps."""

    def __init__(
        self,
        stage_caps: dict[str, dict[str, int]],
        *,
        ladder: list[str] | None = None,
        truncation_factor: float = 0.5,
    ):
        self.stage_caps = stage_caps
        self.ladder = ladder or [RUNG_REDUCE_TRUNCATION, RUNG_DROP_TIER, RUNG_LIMIT_FILINGS]
        self.truncation_factor = truncation_factor
        self._engaged = 0  # number of ladder rungs engaged so far this run

    def evaluate(self, stage: str, run_spend_tokens: int) -> BudgetDecision:
        caps = self.stage_caps.get(stage, {})
        soft = caps.get("soft_cap")
        hard = caps.get("hard_cap")
        exhausted = hard is not None and run_spend_tokens >= hard
        if soft is not None and run_spend_tokens >= soft:
            self._engaged = min(len(self.ladder), self._engaged + 1)
        return BudgetDecision(
            degradations=list(self.ladder[: self._engaged]), budget_exhausted=exhausted
        )

    @classmethod
    def from_config(cls, budgets: dict) -> BudgetController:
        degr = budgets.get("degradation", {})
        return cls(
            budgets.get("stages", {}),
            ladder=degr.get("ladder"),
            truncation_factor=float(degr.get("reduced_truncation_factor", 0.5)),
        )


def next_cheaper_tier(model: str, tier_order: list[str]) -> str:
    """Return the next-cheaper model in the (cheapest-first) tier order; clamp at cheapest."""
    if model not in tier_order:
        return model
    idx = tier_order.index(model)
    return tier_order[max(0, idx - 1)]


@dataclass
class StagePlan:
    """The concrete knobs to run a stage with, after applying the active degradations."""

    model: str
    max_section_chars: int
    max_filings: int | None
    degradations: list[str]
    budget_exhausted: bool


def plan_stage(
    *,
    decision: BudgetDecision,
    model: str,
    base_max_section_chars: int,
    tier_order: list[str],
    truncation_factor: float,
) -> StagePlan:
    """Translate a BudgetDecision into concrete stage knobs."""
    max_section_chars = base_max_section_chars
    max_filings: int | None = None
    for rung in decision.degradations:
        if rung == RUNG_REDUCE_TRUNCATION:
            max_section_chars = max(1000, int(max_section_chars * truncation_factor))
        elif rung == RUNG_DROP_TIER:
            model = next_cheaper_tier(model, tier_order)
        elif rung == RUNG_LIMIT_FILINGS:
            max_filings = 2
    return StagePlan(
        model=model,
        max_section_chars=max_section_chars,
        max_filings=max_filings,
        degradations=list(decision.degradations),
        budget_exhausted=decision.budget_exhausted,
    )
