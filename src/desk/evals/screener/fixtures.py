"""Load the synthetic screener fixture universe into a :class:`MetricsSource`."""

from __future__ import annotations

from pathlib import Path

import yaml

from desk.data.metrics import Metrics
from desk.data.metrics_source import InMemoryMetricsSource

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "screener" / "universe_fixture.yaml"


def _f(value) -> float | None:
    # Coerce numeric fields to float. PyYAML parses sign-less scientific notation like "5.0e9"
    # as a string, so accept both str and number here.
    if value is None:
        return None
    return float(value)


def _row_to_metrics(row: dict) -> Metrics:
    return Metrics(
        ticker=str(row["ticker"]).upper(),
        cik=f"{abs(hash(row['ticker'])) % 10**10:010d}",
        company_name=row.get("company_name", ""),
        sector=row.get("sector"),
        industry=row.get("industry"),
        market_cap=_f(row.get("market_cap")),
        trailing_pe=_f(row.get("trailing_pe")),
        revenue_ttm=_f(row.get("revenue_ttm")),
        revenue_growth_yoy=_f(row.get("revenue_growth_yoy")),
        gross_margin=_f(row.get("gross_margin")),
        operating_margin=_f(row.get("operating_margin")),
        gross_margin_trend=_f(row.get("gross_margin_trend")),
        operating_margin_trend=_f(row.get("operating_margin_trend")),
        net_debt_to_ebitda=_f(row.get("net_debt_to_ebitda")),
        fiscal_year=2025,
    )


def load_fixture_source(path: Path | str = DEFAULT_FIXTURE) -> InMemoryMetricsSource:
    data = yaml.safe_load(Path(path).read_text("utf-8"))
    rows = [_row_to_metrics(r) for r in data.get("tickers", [])]
    return InMemoryMetricsSource(rows)


def fixture_tickers(path: Path | str = DEFAULT_FIXTURE) -> set[str]:
    return {r.ticker for r in load_fixture_source(path).get_table()}
