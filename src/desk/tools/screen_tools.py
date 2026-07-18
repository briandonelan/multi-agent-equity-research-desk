"""Screener-only MCP server: ``run_screen`` and ``get_metrics``.

The ``*_logic`` functions are pure over the data layer and unit-testable without the SDK. The
metrics table is supplied via a :class:`~desk.data.metrics_source.MetricsSource` so the same
screener agent can run against the live universe or a synthetic fixture; ``build_server`` binds
one source into the tool handlers. The ``@tool`` wrappers just call the logic and format the
result (origin + timestamp stamped by ``tool_result``).
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from desk.data import universe
from desk.data.metrics_source import MetricsSource, ProductionMetricsSource
from desk.tools.util import error_result, tool_result

SCREEN_ORIGIN = "edgar+yfinance"


def run_screen_logic(
    filters: list[dict[str, Any]],
    *,
    sort: str | None = None,
    descending: bool = True,
    limit: int = 10,
    source: MetricsSource | None = None,
) -> dict[str, Any]:
    src = source or ProductionMetricsSource()
    rows = universe.run_screen(
        filters, sort=sort, descending=descending, limit=limit, rows=src.get_table()
    )
    return {"origin": SCREEN_ORIGIN, "count": len(rows), "rows": rows}


def get_metrics_logic(ticker: str, *, source: MetricsSource | None = None) -> dict[str, Any]:
    src = source or ProductionMetricsSource()
    m = src.get_row(ticker)
    if m is None:
        return {"origin": SCREEN_ORIGIN, "error": f"{ticker} not in universe"}
    return {"origin": SCREEN_ORIGIN, "metrics": m.as_dict()}


_RUN_SCREEN_SCHEMA = {
    "type": "object",
    "properties": {
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "op": {"type": "string"},
                    "value": {},
                },
                "required": ["field", "op", "value"],
            },
        },
        "sort": {"type": "string"},
        "descending": {"type": "boolean"},
        "limit": {"type": "integer"},
    },
    "required": ["filters"],
}

_GET_METRICS_SCHEMA = {
    "type": "object",
    "properties": {"ticker": {"type": "string"}},
    "required": ["ticker"],
}


def build_server(source: MetricsSource | None = None):
    """Build the screener's MCP server bound to a metrics source (default: production universe).

    Surfaces as mcp__screen__run_screen / mcp__screen__get_metrics.
    """
    src = source or ProductionMetricsSource()

    @tool("run_screen", "Run a declarative screen over the metrics universe.", _RUN_SCREEN_SCHEMA)
    async def _run_screen(args: dict) -> dict:
        try:
            return tool_result(
                run_screen_logic(
                    args.get("filters", []),
                    sort=args.get("sort"),
                    descending=args.get("descending", True),
                    limit=int(args.get("limit", 10)),
                    source=src,
                )
            )
        except ValueError as exc:
            return error_result(str(exc))

    @tool("get_metrics", "Get the derived metrics row for a single ticker.", _GET_METRICS_SCHEMA)
    async def _get_metrics(args: dict) -> dict:
        return tool_result(get_metrics_logic(str(args.get("ticker", "")).upper(), source=src))

    return create_sdk_mcp_server("screen", tools=[_run_screen, _get_metrics])


TOOL_NAMES = ["mcp__screen__run_screen", "mcp__screen__get_metrics"]
