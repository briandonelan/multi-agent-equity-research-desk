"""Boundary semantic validation: citation resolution, quote matching, ticker universe, JSON."""

from __future__ import annotations

from desk.agents.base import (
    extract_json,
    quote_matches,
    resolve_passage,
    validate_citations,
    validate_tickers_in_universe,
)
from desk.contracts.v1 import Citation, Claim, ThesisDraft
from desk.data import sections


def _seed_section():
    # Two passages separated by the blank-line delimiter used by Section.text.
    text = (
        "Gross margin expanded to 47 percent from 44 percent driven by favorable mix."
        "\n\n"
        "Management believes the improvement is sustainable given ongoing cost discipline."
    )
    sections.store_section_text("ACC-9", "7", text)


def test_resolve_passage_by_index():
    _seed_section()
    p0 = resolve_passage("ACC-9#7¶0")
    p1 = resolve_passage("ACC-9#7¶1")
    assert p0 and "Gross margin expanded" in p0
    assert p1 and "sustainable" in p1
    assert resolve_passage("MISSING#7¶0") is None


def test_quote_matches_fuzzy_and_whitespace():
    passage = "Gross margin expanded to 47 percent from 44 percent driven by favorable mix."
    assert quote_matches("gross   margin expanded to 47 percent", passage)
    assert not quote_matches("revenue declined sharply this year overall", passage)


def _thesis_with_citation(source_ref, quote):
    c = Claim(text="t", claim_type="fact", citations=[Citation(source_ref=source_ref, quote=quote)])
    return ThesisDraft(run_id="r", ticker="CAT", thesis_summary="s", claims=[c, c, c, c])


def test_validate_citations_pass_and_fail():
    _seed_section()
    ok = _thesis_with_citation("ACC-9#7¶0", "Gross margin expanded to 47 percent")
    assert validate_citations(ok) == []

    bad_ref = _thesis_with_citation("NOPE#7¶0", "whatever")
    assert validate_citations(bad_ref)  # unresolvable ref -> error

    bad_quote = _thesis_with_citation("ACC-9#7¶0", "totally fabricated unrelated sentence here")
    assert validate_citations(bad_quote)  # quote not in passage -> error


def test_validate_tickers_in_universe():
    good = ThesisDraft(
        run_id="r",
        ticker="CAT",
        thesis_summary="s",
        claims=_thesis_with_citation("ACC-9#7¶0", "x").claims,
    )
    _seed_section()
    assert validate_tickers_in_universe(good) == []
    bad = ThesisDraft(run_id="r", ticker="ZZZZ", thesis_summary="s", claims=good.claims)
    assert validate_tickers_in_universe(bad)


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('Here is the memo: {"a": 3} thanks') == {"a": 3}
    assert extract_json("not json at all") is None
