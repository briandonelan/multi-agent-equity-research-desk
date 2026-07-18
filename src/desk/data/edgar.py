"""SEC EDGAR client: rate-limited, disk-cached access to the free ``data.sec.gov`` APIs.

Endpoints wrapped:

- ``company_tickers`` — ticker -> CIK map
- ``submissions`` — recent-filings index per company
- ``companyfacts`` — all XBRL facts (revenue, income, margins, debt, shares)
- filing documents — primary 10-K/10-Q/8-K HTML

Design notes:

- A process-global rate limiter keeps us at <= 8 requests/second (SEC's fair-access limit is
  10/s; we leave headroom).
- Every response is disk-cached (see ``cache.py``) so repeated runs are near-zero network.
- ``_http_get`` is the only network call site, so ``respx`` can mock it in tests while the
  caching/parse logic runs unchanged.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import httpx

from desk.data import cache
from desk.settings import get_settings

# --- Rate limiting --------------------------------------------------------------------------

_MAX_PER_SECOND = 8
_lock = threading.Lock()
_recent: deque[float] = deque()


def _rate_limit() -> None:
    """Block until issuing another request stays within the per-second budget."""
    with _lock:
        now = time.monotonic()
        while _recent and now - _recent[0] >= 1.0:
            _recent.popleft()
        if len(_recent) >= _MAX_PER_SECOND:
            sleep_for = 1.0 - (now - _recent[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            while _recent and now - _recent[0] >= 1.0:
                _recent.popleft()
        _recent.append(time.monotonic())


# --- HTTP -----------------------------------------------------------------------------------


class EdgarError(RuntimeError):
    """Raised when EDGAR returns a non-200 or the response can't be parsed."""


def _headers() -> dict[str, str]:
    return {
        "User-Agent": get_settings().sec_edgar_user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


def _http_get(url: str, *, timeout: float = 30.0) -> httpx.Response:
    """Single network call site. Rate-limited; raises EdgarError on transport failure."""
    _rate_limit()
    try:
        resp = httpx.get(url, headers=_headers(), timeout=timeout, follow_redirects=True)
    except httpx.HTTPError as exc:  # pragma: no cover - network error path
        raise EdgarError(f"EDGAR request failed: {url}: {exc}") from exc
    if resp.status_code != 200:
        raise EdgarError(f"EDGAR returned {resp.status_code} for {url}")
    return resp


def cik_to_str(cik: int | str) -> str:
    """Normalize a CIK to the zero-padded 10-digit string EDGAR uses in paths."""
    return f"{int(cik):010d}"


# --- Public endpoint wrappers ---------------------------------------------------------------


def company_tickers(*, force: bool = False) -> dict:
    """The full ticker -> CIK map. Cached 7 days."""
    key = "company_tickers"
    if not force:
        cached = cache.get_json("company_tickers", key)
        if cached is not None:
            return cached
    resp = _http_get("https://www.sec.gov/files/company_tickers.json")
    data = resp.json()
    cache.set_json("company_tickers", key, data, origin="edgar")
    return data


def submissions(cik: int | str, *, force: bool = False) -> dict:
    """Recent-filings index for a company. Cached 1 day."""
    cik10 = cik_to_str(cik)
    key = f"CIK{cik10}"
    if not force:
        cached = cache.get_json("submissions", key)
        if cached is not None:
            return cached
    resp = _http_get(f"https://data.sec.gov/submissions/CIK{cik10}.json")
    data = resp.json()
    cache.set_json("submissions", key, data, origin="edgar")
    return data


def company_facts(cik: int | str, *, force: bool = False) -> dict:
    """All XBRL facts for a company. Cached 1 day."""
    cik10 = cik_to_str(cik)
    key = f"CIK{cik10}"
    if not force:
        cached = cache.get_json("companyfacts", key)
        if cached is not None:
            return cached
    resp = _http_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")
    data = resp.json()
    cache.set_json("companyfacts", key, data, origin="edgar")
    return data


def _accession_nodash(accession: str) -> str:
    return accession.replace("-", "")


def filing_document(cik: int | str, accession: str, filename: str, *, force: bool = False) -> str:
    """Fetch a primary filing document's raw text. Content-addressed, cached forever."""
    cik_int = int(cik)
    acc_nodash = _accession_nodash(accession)
    key = f"{cik_int}/{acc_nodash}/{filename}"
    if not force:
        cached = cache.load_blob("filing", key)
        if cached is not None:
            return cached
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{filename}"
    resp = _http_get(url)
    text = resp.text
    cache.store_blob("filing", key, text)
    return text
