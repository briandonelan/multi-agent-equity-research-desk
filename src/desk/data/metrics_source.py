"""A small seam between the screen tools and the metrics table they read.

Production reads the cached real universe; evaluation harnesses read a committed synthetic
fixture. Both satisfy :class:`MetricsSource`, so the screener agent can be run for real against
either table without touching production code paths.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from desk.data.metrics import Metrics


@runtime_checkable
class MetricsSource(Protocol):
    """Where the screen tools get their rows from."""

    def get_table(self) -> list[Metrics]: ...

    def get_row(self, ticker: str) -> Metrics | None: ...


class ProductionMetricsSource:
    """The default source: the cached universe built from EDGAR + yfinance."""

    def get_table(self) -> list[Metrics]:
        from desk.data import universe

        return universe.load_table()

    def get_row(self, ticker: str) -> Metrics | None:
        from desk.data import universe

        return universe.get_metrics(ticker)


class InMemoryMetricsSource:
    """A source backed by an explicit list of rows — used by the eval fixtures and tests."""

    def __init__(self, rows: list[Metrics]):
        self._rows = list(rows)
        self._by_ticker = {r.ticker.upper(): r for r in rows}

    def get_table(self) -> list[Metrics]:
        return list(self._rows)

    def get_row(self, ticker: str) -> Metrics | None:
        return self._by_ticker.get(ticker.upper())
