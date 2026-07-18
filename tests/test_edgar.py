"""EDGAR client: fixture-backed fetch + caching (second call hits no network)."""

from __future__ import annotations

from desk.data import edgar, filings, ticker_map


def test_company_tickers_and_cache(mock_edgar):
    route = mock_edgar.routes[0]  # company_tickers route
    data = edgar.company_tickers()
    assert any(row["ticker"] == "AAPL" for row in data.values())
    # Second call is served from disk cache — no additional network request.
    edgar.company_tickers()
    assert route.call_count == 1


def test_cik_padding():
    assert edgar.cik_to_str(320193) == "0000320193"
    assert edgar.cik_to_str("320193") == "0000320193"


def test_ticker_map_resolve(mock_edgar):
    company = ticker_map.require("aapl")
    assert company.cik == "0000320193"
    assert "Apple" in company.name
    assert ticker_map.resolve("NOTATICKER") is None


def test_list_filings_forms_and_limit(mock_edgar):
    tenks = filings.list_filings("0000320193", forms=("10-K",), limit=5)
    assert tenks and all(f.form == "10-K" for f in tenks)
    assert tenks[0].primary_document.endswith(".htm")

    latest_10q = filings.latest("0000320193", "10-Q")
    assert latest_10q is not None and latest_10q.form == "10-Q"
