"""Shared pytest fixtures. All tests run offline: network is mocked with ``respx`` and the
cache is redirected to a per-test temp directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


def load_fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text("utf-8")


def load_fixture_json(name: str) -> dict:
    return json.loads(load_fixture_text(name))


@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    """Redirect the disk cache to a temp dir and reset cached settings/config for each test."""
    monkeypatch.setenv("DESK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("DESK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "test-suite test@example.com")

    from desk import settings
    from desk.data import ticker_map

    settings.reset_caches()
    ticker_map.reset_cache()
    yield tmp_path
    settings.reset_caches()
    ticker_map.reset_cache()


@pytest.fixture
def mock_edgar(respx_mock):
    """Route the EDGAR endpoints used by the data layer to committed fixtures."""
    import respx

    cik = "0000320193"
    respx_mock.get("https://www.sec.gov/files/company_tickers.json").respond(
        json=load_fixture_json("company_tickers.json")
    )
    respx_mock.get(f"https://data.sec.gov/submissions/CIK{cik}.json").respond(
        json=load_fixture_json(f"submissions_CIK{cik}.json")
    )
    respx_mock.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json").respond(
        json=load_fixture_json(f"companyfacts_CIK{cik}.json")
    )
    # Primary 10-K document -> synthetic HTML.
    respx_mock.get(
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
    ).respond(text=load_fixture_text("synthetic_10k.html"))
    return respx.mock
