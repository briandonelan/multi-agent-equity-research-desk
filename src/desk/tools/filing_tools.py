"""Reader/critic MCP server: ``list_filings``, ``get_section``, ``get_xbrl_facts``.

Pure ``*_logic`` functions over the data layer; ``@tool`` wrappers format results. ``get_section``
truncation honors the per-stage limit from the tool context (so the degradation ladder can
shrink it), and always marks truncation explicitly.
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from desk.data import edgar, filings, sections, ticker_map
from desk.tools.util import error_result, get_context, tool_result

EDGAR_ORIGIN = "edgar"


def list_filings_logic(
    ticker: str,
    *,
    forms: list[str] | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    company = ticker_map.resolve(ticker)
    if company is None:
        return {"origin": EDGAR_ORIGIN, "error": f"{ticker} not found"}
    forms_t = tuple(forms) if forms else filings.DEFAULT_FORMS
    recs = filings.list_filings(company.cik, forms=forms_t, limit=limit)
    return {
        "origin": EDGAR_ORIGIN,
        "ticker": company.ticker,
        "cik": company.cik,
        "filings": [
            {
                "accession": r.accession,
                "form": r.form,
                "filing_date": r.filing_date,
                "report_date": r.report_date,
                "primary_document": r.primary_document,
            }
            for r in recs
        ],
    }


def get_section_logic(
    ticker: str,
    accession: str,
    item: str,
    *,
    max_chars: int | None = None,
) -> dict[str, Any]:
    try:
        section = sections.get_section(ticker, accession, item, max_chars_per_item=max_chars)
    except KeyError as exc:
        return {"origin": EDGAR_ORIGIN, "error": str(exc)}
    if section is None:
        return {"origin": EDGAR_ORIGIN, "error": f"Item {item} not found in {accession}"}
    return {
        "origin": EDGAR_ORIGIN,
        "ticker": ticker.upper(),
        "accession": accession,
        "item": section.item,
        "title": section.title,
        "truncated": section.truncated,
        "source_refs": [p.source_ref for p in section.passages],
        "passages": [{"source_ref": p.source_ref, "text": p.text} for p in section.passages],
    }


def get_xbrl_facts_logic(ticker: str, concepts: list[str]) -> dict[str, Any]:
    company = ticker_map.resolve(ticker)
    if company is None:
        return {"origin": EDGAR_ORIGIN, "error": f"{ticker} not found"}
    facts = edgar.company_facts(company.cik)
    gaap = facts.get("facts", {}).get("us-gaap", {})
    out: dict[str, list[dict]] = {}
    for concept in concepts:
        node = gaap.get(concept)
        if not node:
            out[concept] = []
            continue
        # Prefer USD; take the most recent ~8 annual + quarterly entries with period labels.
        entries = node.get("units", {}).get("USD") or next(iter(node.get("units", {}).values()), [])
        labeled = [
            {
                "value": e.get("val"),
                "start": e.get("start"),
                "end": e.get("end"),
                "fy": e.get("fy"),
                "fp": e.get("fp"),
                "form": e.get("form"),
            }
            for e in entries[-8:]
        ]
        out[concept] = labeled
    return {"origin": EDGAR_ORIGIN, "ticker": company.ticker, "facts": out}


# --- @tool wrappers -------------------------------------------------------------------------


@tool(
    "list_filings",
    "List recent SEC filings (10-K/10-Q/8-K) for a ticker.",
    {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "forms": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer"},
        },
        "required": ["ticker"],
    },
)
async def list_filings_tool(args: dict) -> dict:
    return tool_result(
        list_filings_logic(
            str(args.get("ticker", "")).upper(),
            forms=args.get("forms"),
            limit=int(args.get("limit", 8)),
        )
    )


@tool(
    "get_section",
    "Get an extracted 10-K/10-Q item (e.g. 1A Risk Factors, 7 MD&A) with citeable source_refs.",
    {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "accession": {"type": "string"},
            "item": {"type": "string"},
        },
        "required": ["ticker", "accession", "item"],
    },
)
async def get_section_tool(args: dict) -> dict:
    ctx = get_context()
    try:
        return tool_result(
            get_section_logic(
                str(args.get("ticker", "")).upper(),
                str(args.get("accession", "")),
                str(args.get("item", "")),
                max_chars=ctx.max_section_chars,
            )
        )
    except Exception as exc:  # noqa: BLE001 - surface as a tool error, never crash the agent
        return error_result(f"get_section failed: {exc}")


@tool(
    "get_xbrl_facts",
    "Get numeric XBRL facts (e.g. Revenues, NetIncomeLoss) with period labels for a ticker.",
    {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "concepts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["ticker", "concepts"],
    },
)
async def get_xbrl_facts_tool(args: dict) -> dict:
    return tool_result(
        get_xbrl_facts_logic(str(args.get("ticker", "")).upper(), args.get("concepts", []))
    )


def build_server():
    """The reader/critic MCP server (mcp__filings__list_filings / __get_section / __get_xbrl_facts)."""
    return create_sdk_mcp_server(
        "filings", tools=[list_filings_tool, get_section_tool, get_xbrl_facts_tool]
    )


TOOL_NAMES = [
    "mcp__filings__list_filings",
    "mcp__filings__get_section",
    "mcp__filings__get_xbrl_facts",
]
