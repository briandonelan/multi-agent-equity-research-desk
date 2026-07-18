"""Metrics computation from XBRL companyfacts (+ patched yfinance)."""

from __future__ import annotations

import pytest

from desk.data import metrics, yf


@pytest.fixture
def patch_yf(monkeypatch):
    monkeypatch.setattr(
        yf,
        "_fetch_info_raw",
        lambda ticker: {
            "marketCap": 3_000_000_000_000,
            "trailingPE": 30.0,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "sharesOutstanding": 15_000_000_000,
            "currentPrice": 200.0,
        },
    )


def test_compute_metrics_from_fixture(mock_edgar, patch_yf):
    m = metrics.compute_metrics("AAPL", "0000320193", "Apple Inc.")
    assert m.sector == "Technology"
    assert m.market_cap == 3_000_000_000_000
    # Revenue TTM should be Apple-scale (hundreds of billions), not a comparative-year collapse.
    assert m.revenue_ttm is not None and m.revenue_ttm > 300e9
    # Margins are fractions in a sane range.
    assert m.gross_margin is not None and 0.2 < m.gross_margin < 0.7
    assert m.operating_margin is not None and 0.1 < m.operating_margin < 0.5
    assert m.fiscal_year is not None


def test_annual_flow_keys_by_period_end_not_fy(mock_edgar, patch_yf):
    # Regression: a 10-K carries 3 comparative years all tagged with one `fy`; keying by the
    # period-end year must recover distinct annual revenues (so YoY growth is meaningful).
    m = metrics.compute_metrics("AAPL", "0000320193", "Apple Inc.")
    assert m.revenue_growth_yoy is not None
    assert -0.5 < m.revenue_growth_yoy < 0.5  # a plausible single-year growth rate


def test_missing_data_is_none_not_fabricated(monkeypatch, mock_edgar):
    monkeypatch.setattr(yf, "_fetch_info_raw", lambda ticker: {})  # yfinance returns nothing
    m = metrics.compute_metrics("AAPL", "0000320193", "Apple Inc.")
    assert m.market_cap is None
    assert m.sector is None
    # XBRL-derived fields still populate from companyfacts.
    assert m.revenue_ttm is not None
