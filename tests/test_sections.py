"""Section extraction: item splitting, source_refs, truncation, and fallbacks."""

from __future__ import annotations

from desk.data import sections
from tests.conftest import load_fixture_text


def _html() -> str:
    return load_fixture_text("synthetic_10k.html")


def test_extracts_target_items_with_source_refs():
    ss = sections.extract_sections(_html(), "ACC-1", "10-K")
    assert ss.warning is None
    assert set(ss.sections) == {"1A", "7"}

    risk = ss.sections["1A"]
    assert "Risk Factors" in risk.title
    assert risk.passages, "expected at least one passage"
    # source_ref format: {accession}#{item}¶{idx}
    assert risk.passages[0].source_ref == "ACC-1#1A¶0"
    assert "competition" in risk.text.lower()

    mdna = ss.sections["7"]
    assert "gross margin" in mdna.text.lower()
    assert mdna.passages[0].source_ref.startswith("ACC-1#7¶")


def test_inline_item_references_do_not_split_sections():
    # Item 1A body mentions "Item 7" and Item 7 body mentions "Item 1A" — inline refs must not
    # create spurious section boundaries.
    ss = sections.extract_sections(_html(), "ACC-1", "10-K")
    # Risk factors should still contain the competition paragraph that precedes "see ... Item 7".
    assert "competitors" in ss.sections["1A"].text.lower()


def test_truncation_marks_and_flags():
    ss = sections.extract_sections(_html(), "ACC-1", "10-K", max_chars_per_item=200)
    risk = ss.sections["1A"]
    assert risk.truncated is True
    assert "[TRUNCATED]" in risk.text


def test_unknown_form_falls_back_to_full_document():
    ss = sections.extract_sections(_html(), "ACC-2", "8-K")
    assert "FULL" in ss.sections
    assert ss.warning is not None


def test_no_items_found_falls_back_with_warning():
    ss = sections.extract_sections(
        "<html><body><p>no items here</p></body></html>", "ACC-3", "10-K"
    )
    assert "FULL" in ss.sections
    assert ss.warning is not None


def test_get_section_offline(mock_edgar):
    # Two items resolved end-to-end through the (mocked) fetch path — the "fixtures for 2" case.
    for item in ("1A", "7"):
        section = sections.get_section("AAPL", "0000320193-25-000079", item)
        assert section is not None
        assert section.passages
        assert section.passages[0].source_ref.startswith(f"0000320193-25-000079#{item}")
