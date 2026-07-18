"""Derived per-ticker metrics from XBRL companyfacts + yfinance.

Metrics: market cap, P/E, revenue TTM, revenue growth YoY, gross/operating margin (level +
recent trend), net debt/EBITDA (best-effort), sector. Missing values are ``None`` — never
fabricated.

Simplification: quarterly XBRL tagging is inconsistent across filers, so we compute margins and the
trend from the most recent **annual** (10-K, FY) periods, which are reliably tagged. The
trend field reports the year-over-year change in the margin level.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from desk.data import cache, edgar, yf

# Alternate XBRL concept names to try, in priority order, for each economic quantity.
_REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
]
_GROSS_PROFIT_CONCEPTS = ["GrossProfit"]
_OPERATING_INCOME_CONCEPTS = ["OperatingIncomeLoss"]
_NET_INCOME_CONCEPTS = ["NetIncomeLoss"]
_COST_OF_REVENUE_CONCEPTS = ["CostOfRevenue", "CostOfGoodsAndServicesSold"]
_DA_CONCEPTS = [
    "DepreciationDepletionAndAmortization",
    "DepreciationAmortizationAndAccretionNet",
    "DepreciationAndAmortization",
]
_DEBT_CONCEPTS = ["LongTermDebtNoncurrent", "LongTermDebt"]
_SHORT_DEBT_CONCEPTS = ["LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent"]
_CASH_CONCEPTS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
]


@dataclass
class Metrics:
    ticker: str
    cik: str
    company_name: str
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    trailing_pe: float | None = None
    revenue_ttm: float | None = None
    revenue_growth_yoy: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    gross_margin_trend: float | None = None  # YoY change in gross margin (pp as fraction)
    operating_margin_trend: float | None = None
    net_debt_to_ebitda: float | None = None
    fiscal_year: int | None = None  # FY of the most recent annual figures used

    def as_dict(self) -> dict:
        return asdict(self)


def _end_year(end: str) -> int | None:
    try:
        return int(end[:4])
    except (TypeError, ValueError):
        return None


def _annual_flow(facts: dict, concepts: list[str], unit: str = "USD") -> dict[int, float]:
    """Return {period_end_year: value} for an annual flow concept.

    Keys by the **period end** year, not XBRL's ``fy`` field. A single 10-K reports the current
    year plus two comparative years, all tagged with the filing's ``fy`` but with distinct
    ``end`` dates — keying by ``fy`` would collapse three real years into one. We keep annual
    (~1-year span, ``fp == "FY"``) entries and, on collisions, prefer the most recently filed.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        node = gaap.get(concept)
        if not node:
            continue
        entries = node.get("units", {}).get(unit)
        if not entries:
            continue
        by_year: dict[int, tuple[str, float]] = {}  # end_year -> (filed, val)
        for e in entries:
            if e.get("fp") != "FY":
                continue
            start, end, val = e.get("start"), e.get("end"), e.get("val")
            if val is None or not start or not end:
                continue
            # Keep genuine annual spans (~a year); excludes stub/YTD periods tagged FY.
            if (_end_year(end) or 0) - (_end_year(start) or 0) > 1:
                continue
            span_days = _rough_days(start, end)
            if span_days is not None and span_days < 300:
                continue
            year = _end_year(end)
            if year is None:
                continue
            filed = e.get("filed", "")
            if year not in by_year or filed > by_year[year][0]:
                by_year[year] = (filed, float(val))
        if by_year:
            return {y: v for y, (_, v) in by_year.items()}
    return {}


def _rough_days(start: str, end: str) -> int | None:
    from datetime import date

    try:
        s = date.fromisoformat(start)
        e = date.fromisoformat(end)
        return (e - s).days
    except (TypeError, ValueError):
        return None


def _latest_instant(facts: dict, concepts: list[str], unit: str = "USD") -> float | None:
    """Return the most recent point-in-time value for an instant concept (e.g. debt, cash)."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        node = gaap.get(concept)
        if not node:
            continue
        entries = node.get("units", {}).get(unit)
        if not entries:
            continue
        best_end = ""
        best_val: float | None = None
        for e in entries:
            end = e.get("end", "")
            val = e.get("val")
            if val is None:
                continue
            if end > best_end:
                best_end = end
                best_val = float(val)
        if best_val is not None:
            return best_val
    return None


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def compute_metrics(ticker: str, cik: str, company_name: str, *, force: bool = False) -> Metrics:
    """Compute (and cache) the derived metrics row for one ticker."""
    if not force:
        cached = cache.get_json("metrics", ticker.upper())
        if cached is not None:
            return Metrics(**cached)

    facts = edgar.company_facts(cik)
    info = yf.get_info(ticker)

    revenue = _annual_flow(facts, _REVENUE_CONCEPTS)
    gross = _annual_flow(facts, _GROSS_PROFIT_CONCEPTS)
    op_income = _annual_flow(facts, _OPERATING_INCOME_CONCEPTS)
    cost_rev = _annual_flow(facts, _COST_OF_REVENUE_CONCEPTS)
    da = _annual_flow(facts, _DA_CONCEPTS)

    years = sorted(revenue.keys())
    fy = years[-1] if years else None
    prev_fy = years[-2] if len(years) >= 2 else None

    rev_ttm = revenue.get(fy) if fy is not None else None
    rev_prev = revenue.get(prev_fy) if prev_fy is not None else None
    rev_growth = _safe_div((rev_ttm - rev_prev), rev_prev) if rev_ttm and rev_prev else None

    # Gross profit may be absent; derive from revenue - cost_of_revenue when possible.
    def gross_for(year: int | None) -> float | None:
        if year is None:
            return None
        if year in gross:
            return gross[year]
        if year in revenue and year in cost_rev:
            return revenue[year] - cost_rev[year]
        return None

    gm = _safe_div(gross_for(fy), rev_ttm)
    gm_prev = _safe_div(gross_for(prev_fy), rev_prev)
    om = _safe_div(op_income.get(fy) if fy else None, rev_ttm)
    om_prev = _safe_div(op_income.get(prev_fy) if prev_fy else None, rev_prev)

    gm_trend = (gm - gm_prev) if (gm is not None and gm_prev is not None) else None
    om_trend = (om - om_prev) if (om is not None and om_prev is not None) else None

    # Net debt / EBITDA (best-effort). EBITDA ~= operating income + D&A for the latest FY.
    ebitda = None
    if fy is not None and fy in op_income:
        ebitda = op_income[fy] + (da.get(fy, 0.0))
    total_debt = (_latest_instant(facts, _DEBT_CONCEPTS) or 0.0) + (
        _latest_instant(facts, _SHORT_DEBT_CONCEPTS) or 0.0
    )
    cash = _latest_instant(facts, _CASH_CONCEPTS)
    net_debt = (total_debt - cash) if cash is not None else None
    net_debt_to_ebitda = _safe_div(net_debt, ebitda)

    metrics = Metrics(
        ticker=ticker.upper(),
        cik=cik,
        company_name=company_name,
        sector=info.sector,
        industry=info.industry,
        market_cap=info.market_cap,
        trailing_pe=info.trailing_pe,
        revenue_ttm=rev_ttm,
        revenue_growth_yoy=rev_growth,
        gross_margin=gm,
        operating_margin=om,
        gross_margin_trend=gm_trend,
        operating_margin_trend=om_trend,
        net_debt_to_ebitda=net_debt_to_ebitda,
        fiscal_year=fy,
    )
    cache.set_json("metrics", ticker.upper(), metrics.as_dict(), origin="edgar+yfinance")
    return metrics
