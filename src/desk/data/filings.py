"""Structured access to a company's filings via the EDGAR submissions index.

Turns the parallel-array ``filings.recent`` shape into a list of :class:`Filing` records and
resolves the primary document URL/filename for each. Pure over the data layer — no network
beyond ``edgar.submissions`` (itself cached).
"""

from __future__ import annotations

from dataclasses import dataclass

from desk.data import edgar

DEFAULT_FORMS = ("10-K", "10-Q", "8-K")


@dataclass(frozen=True)
class Filing:
    cik: str  # zero-padded 10-digit
    accession: str  # dashed form, e.g. 0000320193-24-000123
    form: str
    filing_date: str  # YYYY-MM-DD
    report_date: str  # period of report, YYYY-MM-DD (may be "")
    primary_document: str  # filename of the primary doc within the accession folder
    primary_doc_description: str


def list_filings(
    cik: int | str,
    *,
    forms: tuple[str, ...] | list[str] = DEFAULT_FORMS,
    limit: int = 8,
) -> list[Filing]:
    """Most-recent-first filings for a company, filtered to the requested form types."""
    wanted = {f.upper() for f in forms}
    subs = edgar.submissions(cik)
    cik10 = edgar.cik_to_str(cik)
    recent = subs.get("filings", {}).get("recent", {})

    accession = recent.get("accessionNumber", [])
    form = recent.get("form", [])
    fdate = recent.get("filingDate", [])
    rdate = recent.get("reportDate", [])
    pdoc = recent.get("primaryDocument", [])
    pdesc = recent.get("primaryDocDescription", [])

    n = len(accession)
    out: list[Filing] = []
    for i in range(n):
        if form[i].upper() not in wanted:
            continue
        out.append(
            Filing(
                cik=cik10,
                accession=accession[i],
                form=form[i],
                filing_date=fdate[i] if i < len(fdate) else "",
                report_date=rdate[i] if i < len(rdate) else "",
                primary_document=pdoc[i] if i < len(pdoc) else "",
                primary_doc_description=pdesc[i] if i < len(pdesc) else "",
            )
        )
        if len(out) >= limit:
            break
    return out


def latest(cik: int | str, form: str) -> Filing | None:
    """The most recent filing of a given form type, or None."""
    filings = list_filings(cik, forms=(form,), limit=1)
    return filings[0] if filings else None
