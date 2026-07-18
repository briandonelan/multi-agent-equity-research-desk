"""MCP tool logic (pure functions) + result formatting."""

from __future__ import annotations

import json

import pytest

from desk.data import yf
from desk.tools import filing_tools, screen_tools
from desk.tools.util import ToolContext, get_context, set_context, tool_result, truncate


@pytest.fixture
def patch_yf(monkeypatch):
    monkeypatch.setattr(
        yf,
        "_fetch_info_raw",
        lambda t: {"marketCap": 4e11, "trailingPE": 45.0, "sector": "Industrials"},
    )


def test_tool_result_stamps_origin_and_timestamp():
    res = tool_result({"origin": "edgar", "x": 1})
    payload = json.loads(res["content"][0]["text"])
    assert payload["origin"] == "edgar"
    assert "retrieved_at" in payload


def test_truncate_marks_explicitly():
    text, was = truncate("abcdefghij", 4)
    assert was is True
    assert text.startswith("abcd") and "[TRUNCATED]" in text
    text2, was2 = truncate("ab", 4)
    assert was2 is False and text2 == "ab"


def test_get_metrics_logic(mock_edgar, patch_yf):
    # AAPL is the ticker with committed companyfacts fixtures.
    out = screen_tools.get_metrics_logic("AAPL")
    assert out["origin"] == "edgar+yfinance"
    assert out["metrics"]["ticker"] == "AAPL"


def test_get_metrics_logic_unknown_ticker():
    out = screen_tools.get_metrics_logic("ZZZZ")
    assert "error" in out


def test_list_filings_logic(mock_edgar):
    out = filing_tools.list_filings_logic("AAPL", forms=["10-K"], limit=3)
    assert out["origin"] == "edgar"
    assert out["filings"] and all(f["form"] == "10-K" for f in out["filings"])


def test_get_section_logic_with_source_refs(mock_edgar):
    out = filing_tools.get_section_logic("AAPL", "0000320193-25-000079", "1A")
    assert out["origin"] == "edgar"
    assert out["source_refs"] and out["source_refs"][0].startswith("0000320193-25-000079#1A")
    assert out["passages"][0]["text"]


def test_get_section_logic_truncation_honors_context(mock_edgar):
    token = set_context(ToolContext(max_section_chars=300))
    try:
        out = filing_tools.get_section_logic(
            "AAPL", "0000320193-25-000079", "1A", max_chars=get_context().max_section_chars
        )
        assert out["truncated"] is True
    finally:
        from desk.tools.util import reset_context

        reset_context(token)


def test_get_xbrl_facts_logic(mock_edgar):
    out = filing_tools.get_xbrl_facts_logic("AAPL", ["NetIncomeLoss", "Nonexistent"])
    assert out["origin"] == "edgar"
    assert out["facts"]["NetIncomeLoss"], "expected some NetIncomeLoss entries"
    assert out["facts"]["Nonexistent"] == []
    first = out["facts"]["NetIncomeLoss"][0]
    assert "value" in first and "end" in first
