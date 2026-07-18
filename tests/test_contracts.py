"""Contract shape/validation and the critic-isolation view."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from desk.contracts.v1 import (
    Citation,
    Claim,
    ScreenResult,
    ThesisDraft,
)


def _claim(text="x") -> Claim:
    return Claim(
        text=text,
        claim_type="fact",
        citations=[Citation(source_ref="ACC#7¶0", quote="a quote")],
    )


def test_for_critic_drops_summary_and_reasoning():
    draft = ThesisDraft(
        run_id="r",
        ticker="CAT",
        thesis_summary="confident bullish summary the critic must not see",
        claims=[_claim(), _claim(), _claim(), _claim()],
    )
    view = draft.for_critic()
    assert not hasattr(view, "thesis_summary")
    assert len(view.claims) == 4
    # The view carries only claims (text + citations), no confidence language.
    dumped = view.model_dump()
    assert "thesis_summary" not in dumped
    assert set(dumped.keys()) == {"run_id", "ticker", "claims"}


def test_claim_requires_at_least_one_citation():
    with pytest.raises(ValidationError):
        Claim(text="x", claim_type="fact", citations=[])


def test_thesis_claim_count_bounds():
    with pytest.raises(ValidationError):
        ThesisDraft(run_id="r", ticker="CAT", thesis_summary="s", claims=[_claim()])  # < 4


def test_screenresult_candidate_bounds():
    with pytest.raises(ValidationError):
        ScreenResult(run_id="r", interpreted_intent="i", filters_applied=[], candidates=[])


def test_screen_filter_accepts_string_numeric_and_tuple_values():
    from desk.contracts.v1 import ScreenFilter

    assert ScreenFilter(field="sector", op="==", value="Industrials").value == "Industrials"
    assert ScreenFilter(field="operating_margin", op=">", value=0).value == 0.0
    assert ScreenFilter(field="market_cap", op="between", value=[2e9, 1e10]).value == (2e9, 1e10)
