"""Screen engine: declarative filters, sort, limit, missing-value + validation behavior."""

from __future__ import annotations

import pytest

from desk.data import universe
from desk.data.metrics import Metrics


def _rows() -> list[Metrics]:
    return [
        Metrics(
            "AAA",
            "1",
            "Alpha",
            sector="Industrials",
            market_cap=5e9,
            operating_margin=0.18,
            revenue_growth_yoy=0.10,
            trailing_pe=15.0,
        ),
        Metrics(
            "BBB",
            "2",
            "Beta",
            sector="Industrials",
            market_cap=20e9,
            operating_margin=0.09,
            revenue_growth_yoy=0.03,
            trailing_pe=25.0,
        ),
        Metrics(
            "CCC",
            "3",
            "Gamma",
            sector="Technology",
            market_cap=100e9,
            operating_margin=0.30,
            revenue_growth_yoy=0.20,
            trailing_pe=40.0,
        ),
        Metrics(
            "DDD",
            "4",
            "Delta",
            sector="Industrials",
            market_cap=8e9,
            operating_margin=None,
            revenue_growth_yoy=0.05,
            trailing_pe=None,
        ),
    ]


def test_numeric_and_sector_filters():
    out = universe.run_screen(
        [
            {"field": "sector", "op": "==", "value": "Industrials"},
            {"field": "operating_margin", "op": ">", "value": 0.10},
        ],
        rows=_rows(),
    )
    assert [r["ticker"] for r in out] == ["AAA"]


def test_between_and_sort_and_limit():
    out = universe.run_screen(
        [{"field": "market_cap", "op": "between", "value": (1e9, 200e9)}],
        sort="market_cap",
        descending=True,
        limit=2,
        rows=_rows(),
    )
    assert [r["ticker"] for r in out] == ["CCC", "BBB"]


def test_missing_value_never_matches_numeric_filter():
    # DDD has operating_margin=None; it must be excluded by a numeric filter, not fabricated.
    out = universe.run_screen(
        [{"field": "operating_margin", "op": ">=", "value": 0.0}], rows=_rows()
    )
    assert "DDD" not in [r["ticker"] for r in out]


def test_limit_is_clamped_to_10():
    out = universe.run_screen(
        [{"field": "market_cap", "op": ">", "value": 0}], limit=999, rows=_rows()
    )
    assert len(out) <= 10


def test_invalid_field_or_op_raises():
    with pytest.raises(ValueError):
        universe.run_screen([{"field": "pe_ratio", "op": ">", "value": 1}], rows=_rows())
    with pytest.raises(ValueError):
        universe.run_screen([{"field": "market_cap", "op": "~", "value": 1}], rows=_rows())


def test_universe_tickers_from_config():
    tickers = universe.universe_tickers()
    assert "AAPL" in tickers
