"""The screening universe and the derived-metrics table it exposes.

The universe is a committed static list (``config/universe.yaml``) so runs are reproducible.
``build`` computes and caches a :class:`~desk.data.metrics.Metrics` row per ticker; after the
first run everything is served from disk. ``run_screen`` is the pure, in-process filter engine
the screener's MCP tool calls — declarative ``{field, op, value}`` filters over the
metrics table, plus sort + limit.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from desk.data import metrics as metrics_mod
from desk.data import ticker_map
from desk.settings import load_yaml_config

# Fields exposed to the screener's declarative filters (numeric + sector).
SCREENABLE_FIELDS = {
    "market_cap",
    "trailing_pe",
    "revenue_ttm",
    "revenue_growth_yoy",
    "gross_margin",
    "operating_margin",
    "gross_margin_trend",
    "operating_margin_trend",
    "net_debt_to_ebitda",
    "sector",
}

_OPS = {"<", "<=", ">", ">=", "==", "between"}


def universe_tickers() -> list[str]:
    cfg = load_yaml_config("universe")
    return [str(t).upper() for t in cfg.get("tickers", [])]


def build(*, force: bool = False, tickers: list[str] | None = None) -> list[metrics_mod.Metrics]:
    """Compute/cache metrics for every ticker in the universe. Idempotent after first run."""
    syms = tickers if tickers is not None else universe_tickers()
    rows: list[metrics_mod.Metrics] = []
    for sym in syms:
        company = ticker_map.resolve(sym)
        if company is None:
            continue
        rows.append(metrics_mod.compute_metrics(sym, company.cik, company.name, force=force))
    return rows


def get_metrics(ticker: str) -> metrics_mod.Metrics | None:
    """Metrics for one ticker (must be in-universe). Computes on miss."""
    company = ticker_map.resolve(ticker)
    if company is None:
        return None
    return metrics_mod.compute_metrics(ticker, company.cik, company.name)


def load_table() -> list[metrics_mod.Metrics]:
    """All universe metrics rows available from cache/compute."""
    rows: list[metrics_mod.Metrics] = []
    for sym in universe_tickers():
        m = get_metrics(sym)
        if m is not None:
            rows.append(m)
    return rows


# --- Screen engine --------------------------------------------------------------------------


def _match(value: Any, op: str, target: Any) -> bool:
    if value is None:
        return False  # missing data never matches a numeric filter
    if op == "==":
        if isinstance(target, str):
            return str(value).lower() == target.lower()
        return value == target
    if op == "between":
        lo, hi = target
        return lo <= value <= hi
    if op == "<":
        return value < target
    if op == "<=":
        return value <= target
    if op == ">":
        return value > target
    if op == ">=":
        return value >= target
    raise ValueError(f"Unknown op: {op}")


def run_screen(
    filters: list[dict[str, Any]],
    *,
    sort: str | None = None,
    descending: bool = True,
    limit: int = 10,
    rows: list[metrics_mod.Metrics] | None = None,
) -> list[dict[str, Any]]:
    """Apply declarative filters to the metrics table; return matching rows as dicts.

    Each filter is ``{"field": str, "op": str, "value": number | [lo, hi] | str}``. Unknown
    fields/ops raise ``ValueError`` so the screener gets a repairable error rather than silent
    wrong results. ``limit`` is clamped to 10.
    """
    limit = max(1, min(int(limit), 10))
    table = rows if rows is not None else load_table()

    valid_names = {f.name for f in fields(metrics_mod.Metrics)}

    for filt in filters:
        field = filt.get("field")
        op = filt.get("op")
        if field not in SCREENABLE_FIELDS:
            raise ValueError(
                f"Field not screenable: {field!r}. Allowed: {sorted(SCREENABLE_FIELDS)}"
            )
        if op not in _OPS:
            raise ValueError(f"Unknown op: {op!r}. Allowed: {sorted(_OPS)}")

    matched: list[metrics_mod.Metrics] = []
    for row in table:
        if all(_match(getattr(row, f["field"]), f["op"], f["value"]) for f in filters):
            matched.append(row)

    if sort is not None and sort in valid_names:
        matched.sort(
            key=lambda r: (getattr(r, sort) is not None, getattr(r, sort) or 0),
            reverse=descending,
        )

    return [r.as_dict() for r in matched[:limit]]
