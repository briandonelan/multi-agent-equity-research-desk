"""Best-effort extraction of 10-K / 10-Q items from primary filing HTML.

We target the items the reader and critic actually cite:

- 10-K: Item 1A (Risk Factors), Item 7 (MD&A)
- 10-Q: Item 2 (MD&A)   [Part I, Item 2 — "Management's Discussion and Analysis"]

SEC HTML is inconsistent, so this is deliberately a heuristic splitter over the flattened
text, not a DOM-structural parse. Every filing appears to list each item twice (table of
contents + body); we keep the **longest** segment for each item id, which is reliably the body.
If we can't find the target items at all, we fall back to whole-document text and set a
``warning`` flag on the result so downstream contracts can surface the degradation.

Each extracted paragraph carries a stable ``source_ref`` of the form
``{accession}#{item}¶{paragraph_idx}`` so memos can cite it and boundary validation can
resolve it back to the cached passage.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from desk.data import cache

# Some SEC primary documents are XML-declared (inline XBRL) but we parse them with the lxml HTML
# parser on purpose (best-effort text flattening). Suppress bs4's advisory warning so it does not
# flood every run's output.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Items we extract, with human titles, per form.
TARGET_ITEMS: dict[str, dict[str, str]] = {
    "10-K": {"1A": "Risk Factors", "7": "Management's Discussion and Analysis"},
    "10-Q": {"2": "Management's Discussion and Analysis"},
}

# Matches an item heading at the START of a line — "Item 1A.", "ITEM 7 —", "Item 2:".
# Line-anchoring is what separates real section headings (on their own line after HTML
# flattening) from inline cross-references like "see Item 7 of this report", which would
# otherwise fragment the sections.
_ITEM_RE = re.compile(r"(?im)^\s*item\s+(\d{1,2}[abc]?)\s*[.:)\-–— ]")

_MIN_SECTION_CHARS = 400  # below this, a "section" is almost certainly a TOC stub
_MIN_PARAGRAPH_CHARS = 40
_TARGET_PARAGRAPH_CHARS = 900  # pack flattened lines into passages of roughly this size


@dataclass(frozen=True)
class Passage:
    source_ref: str
    text: str


@dataclass
class Section:
    item: str
    title: str
    passages: list[Passage] = field(default_factory=list)
    truncated: bool = False

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.passages)


@dataclass
class SectionSet:
    accession: str
    form: str
    sections: dict[str, Section] = field(default_factory=dict)
    warning: str | None = None


def _flatten_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    # Normalize unicode spaces and collapse whitespace while preserving line breaks.
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(lines)


def _split_paragraphs(text: str) -> list[str]:
    """Reconstruct citeable passages from flattened text.

    SEC HTML flattening is inconsistent: sometimes each paragraph is its own line, sometimes a
    whole section is one line. So rather than trust blank-line structure, we split on any
    newline and greedily **pack** lines into passages of roughly ``_TARGET_PARAGRAPH_CHARS``,
    breaking only at line boundaries. This yields stable, similarly-sized passages (good for
    ``source_ref`` granularity) regardless of the source formatting.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    passages: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for line in lines:
        buf.append(line)
        buf_len += len(line) + 1
        if buf_len >= _TARGET_PARAGRAPH_CHARS:
            passages.append(" ".join(buf).strip())
            buf, buf_len = [], 0
    if buf:
        tail = " ".join(buf).strip()
        if len(tail) >= _MIN_PARAGRAPH_CHARS or not passages:
            passages.append(tail)
        elif passages:
            passages[-1] = f"{passages[-1]} {tail}".strip()
    return [p for p in passages if p]


def _segment_by_item(text: str) -> dict[str, str]:
    """Return {item_id: longest_body_segment} across all item markers found in the text."""
    markers = [(m.group(1).upper(), m.start()) for m in _ITEM_RE.finditer(text)]
    if not markers:
        return {}
    best: dict[str, str] = {}
    for idx, (item_id, start) in enumerate(markers):
        end = markers[idx + 1][1] if idx + 1 < len(markers) else len(text)
        segment = text[start:end]
        if item_id not in best or len(segment) > len(best[item_id]):
            best[item_id] = segment
    return best


def extract_sections(
    html: str,
    accession: str,
    form: str,
    *,
    max_chars_per_item: int | None = None,
) -> SectionSet:
    """Extract the target items for the filing form into a :class:`SectionSet`."""
    form = form.upper()
    targets = TARGET_ITEMS.get(form)
    result = SectionSet(accession=accession, form=form)

    text = _flatten_html(html)

    if targets is None:
        # Non 10-K/10-Q (e.g. 8-K): expose whole-document text under item "FULL".
        result.warning = f"No item map for form {form}; using whole-document text."
        result.sections["FULL"] = _build_section(
            "FULL", "Full Document", text, accession, max_chars_per_item
        )
        return result

    segments = _segment_by_item(text)
    found_any = False
    for item_id, title in targets.items():
        body = segments.get(item_id)
        if body and len(body) >= _MIN_SECTION_CHARS:
            result.sections[item_id] = _build_section(
                item_id, title, body, accession, max_chars_per_item
            )
            found_any = True

    if not found_any:
        result.warning = (
            f"Could not locate target items {list(targets)} in {accession}; "
            "using whole-document text."
        )
        result.sections["FULL"] = _build_section(
            "FULL", "Full Document", text, accession, max_chars_per_item
        )

    return result


def _build_section(
    item_id: str,
    title: str,
    body: str,
    accession: str,
    max_chars: int | None,
) -> Section:
    truncated = False
    if max_chars is not None and len(body) > max_chars:
        body = body[:max_chars] + "\n\n[TRUNCATED]"
        truncated = True
    paragraphs = _split_paragraphs(body)
    passages = [
        Passage(source_ref=f"{accession}#{item_id}¶{i}", text=p) for i, p in enumerate(paragraphs)
    ]
    return Section(item=item_id, title=title, passages=passages, truncated=truncated)


# --- Caching of extracted sections ----------------------------------------------------------


def _cache_key(accession: str, item: str) -> str:
    return f"{accession}/{item}"


def store_section_text(accession: str, item: str, text: str) -> None:
    cache.store_blob("section", _cache_key(accession, item), text)


def load_section_text(accession: str, item: str) -> str | None:
    return cache.load_blob("section", _cache_key(accession, item))


def persist(section_set: SectionSet) -> None:
    """Write every extracted section's text to the content-addressed cache."""
    for item, section in section_set.sections.items():
        store_section_text(section_set.accession, item, section.text)


# --- High-level accessors (fetch + extract) -------------------------------------------------


def fetch_and_extract(
    ticker: str,
    accession: str,
    *,
    max_chars_per_item: int | None = None,
) -> SectionSet:
    """Resolve a filing for ``ticker``/``accession``, fetch its primary doc, and extract items."""
    from desk.data import edgar, filings, ticker_map

    company = ticker_map.require(ticker)
    # Find the filing record (any of the tracked forms) to get its form + primary document.
    record = next(
        (f for f in filings.list_filings(company.cik, limit=40) if f.accession == accession),
        None,
    )
    if record is None:
        raise KeyError(f"Accession {accession!r} not found among recent filings for {ticker}")

    html = edgar.filing_document(company.cik, accession, record.primary_document)
    section_set = extract_sections(
        html, accession, record.form, max_chars_per_item=max_chars_per_item
    )
    persist(section_set)
    return section_set


def get_section(
    ticker: str,
    accession: str,
    item: str,
    *,
    max_chars_per_item: int | None = None,
) -> Section | None:
    """Return one extracted :class:`Section` (with source_refs), or None if absent."""
    section_set = fetch_and_extract(ticker, accession, max_chars_per_item=max_chars_per_item)
    item = item.upper()
    if item in section_set.sections:
        return section_set.sections[item]
    # If extraction fell back to whole-document text, expose that instead of nothing.
    return section_set.sections.get("FULL")
