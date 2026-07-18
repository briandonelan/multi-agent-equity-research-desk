"""yfinance wrapper — treated as unreliable, so wrapped with retry + disk cache.

yfinance is the only source for market cap / trailing P/E / sector. We cache the
subset of ``Ticker.info`` we need per ticker per day. ``_fetch_info_raw`` is the sole network
call site so tests can monkeypatch it (or pre-seed the cache) and stay offline.

Missing values are ``None``, never fabricated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from desk.data import cache

# Fields we pull from yfinance's info dict, mapped to our normalized keys.
_INFO_FIELDS = {
    "market_cap": "marketCap",
    "trailing_pe": "trailingPE",
    "forward_pe": "forwardPE",
    "sector": "sector",
    "industry": "industry",
    "shares_outstanding": "sharesOutstanding",
    "price": "currentPrice",
}


@dataclass(frozen=True)
class YfInfo:
    ticker: str
    market_cap: float | None
    trailing_pe: float | None
    forward_pe: float | None
    sector: str | None
    industry: str | None
    shares_outstanding: float | None
    price: float | None


def _fetch_info_raw(ticker: str) -> dict:
    """Network call to yfinance. Isolated so tests can patch it."""
    import yfinance  # imported lazily; heavy dependency

    return dict(yfinance.Ticker(ticker).info or {})


def _fetch_with_retry(ticker: str, *, attempts: int = 3, base_delay: float = 1.0) -> dict:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            info = _fetch_info_raw(ticker)
            if info:
                return info
        except Exception as exc:  # yfinance raises a variety of errors; treat all as retryable
            last_exc = exc
        time.sleep(base_delay * (2**attempt))
    if last_exc is not None:
        # Exhausted retries with errors: surface an empty dict rather than crash the run.
        return {}
    return {}


def get_info(ticker: str, *, force: bool = False) -> YfInfo:
    """Return normalized info for a ticker. Cached 1 day; missing fields are None."""
    ticker = ticker.upper().strip()
    if not force:
        cached = cache.get_json("yfinance", ticker)
        if cached is not None:
            return _to_info(ticker, cached)

    raw = _fetch_with_retry(ticker)
    subset = {norm: raw.get(src) for norm, src in _INFO_FIELDS.items()}
    cache.set_json("yfinance", ticker, subset, origin="yfinance")
    return _to_info(ticker, subset)


def _num(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_info(ticker: str, subset: dict) -> YfInfo:
    return YfInfo(
        ticker=ticker,
        market_cap=_num(subset.get("market_cap")),
        trailing_pe=_num(subset.get("trailing_pe")),
        forward_pe=_num(subset.get("forward_pe")),
        sector=subset.get("sector") or None,
        industry=subset.get("industry") or None,
        shares_outstanding=_num(subset.get("shares_outstanding")),
        price=_num(subset.get("price")),
    )
