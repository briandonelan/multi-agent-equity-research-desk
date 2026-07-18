"""Ticker <-> CIK resolution built on the SEC ``company_tickers.json`` map.

The map is normalized once into ``{ticker: (cik, company_name)}`` and cached in-process. All
lookups are case-insensitive on the ticker.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from desk.data import edgar


@dataclass(frozen=True)
class Company:
    ticker: str
    cik: str  # zero-padded 10-digit
    name: str


@functools.lru_cache(maxsize=1)
def _index() -> dict[str, Company]:
    raw = edgar.company_tickers()
    out: dict[str, Company] = {}
    # EDGAR shape: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    for row in raw.values():
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        out[ticker] = Company(
            ticker=ticker,
            cik=edgar.cik_to_str(row["cik_str"]),
            name=str(row.get("title", "")).strip(),
        )
    return out


def reset_cache() -> None:
    _index.cache_clear()


def resolve(ticker: str) -> Company | None:
    """Return the Company for a ticker, or None if it is not in the SEC map."""
    return _index().get(ticker.upper().strip())


def require(ticker: str) -> Company:
    company = resolve(ticker)
    if company is None:
        raise KeyError(f"Ticker not found in SEC company map: {ticker!r}")
    return company


def all_companies() -> list[Company]:
    return list(_index().values())
