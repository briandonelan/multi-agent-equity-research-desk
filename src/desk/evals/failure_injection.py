"""Handoff failure-injection experiments.

Config-driven chaos at the handoff boundary. For each fault we corrupt a valid artifact (or tool
config), run the SAME boundary validation the orchestrator uses, and report whether it was caught
(by Pydantic shape or by semantic validation) — the raw material for a failure taxonomy. Injected
runs are tagged in the ledger via the ``injected_fault`` column.

Faults:
- truncate_citations: strip a claim's citations before the critic (Pydantic min_length catch).
- schema_drift:       rename a required field (Pydantic missing-field catch).
- ambiguous_ticker:   put an out-of-universe alias in a screener candidate (semantic catch).
- tool_truncation:    force get_section truncation to 25% (degraded evidence, marked explicitly).
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from desk.agents.base import validate_tickers_in_universe
from desk.contracts.v1 import Artifact, ScreenResult, ThesisDraft
from desk.ledger import db
from desk.tools.util import DEFAULT_MAX_SECTION_CHARS

FAULTS = ["truncate_citations", "schema_drift", "ambiguous_ticker", "tool_truncation"]


@dataclass
class InjectionResult:
    fault: str
    description: str
    caught_by: str  # "pydantic" | "semantic" | "none" | "tool"
    errors: list[str]

    def as_dict(self) -> dict:
        return {
            "fault": self.fault,
            "description": self.description,
            "caught_by": self.caught_by,
            "errors": self.errors,
        }


def _valid_thesis_payload() -> dict:
    cite = {"source_ref": "ACC-INJ#7¶0", "quote": "Revenue increased 4 percent year over year"}
    claim = {"text": "Revenue grew.", "claim_type": "fact", "citations": [cite]}
    return {
        "run_id": "INJ",
        "ticker": "CAT",
        "thesis_summary": "s",
        "claims": [claim, claim, claim, claim],
    }


def _valid_screen_payload() -> dict:
    return {
        "run_id": "INJ",
        "interpreted_intent": "i",
        "filters_applied": [],
        "candidates": [
            {
                "ticker": "CAT",
                "cik": "1",
                "company_name": "Caterpillar Inc.",
                "rationale": "r",
                "metrics_snapshot": {},
            }
        ],
    }


def _boundary_check(
    payload: dict, contract_cls: type[Artifact], validators
) -> tuple[str, list[str]]:
    """Mirror the orchestrator's boundary validation (no LLM). Returns (caught_by, errors)."""
    try:
        artifact = contract_cls.model_validate(payload)
    except ValidationError as exc:
        return "pydantic", [
            f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()
        ]
    errors: list[str] = []
    for v in validators:
        errors.extend(v(artifact))
    if errors:
        return "semantic", errors
    return "none", []


def run_injection(fault: str, *, run_id: str = "INJECT") -> InjectionResult:
    """Apply one fault and report whether boundary validation caught it. Tags the ledger."""
    if fault == "truncate_citations":
        payload = _valid_thesis_payload()
        for claim in payload["claims"]:
            claim["citations"] = []  # strip citations
        caught_by, errors = _boundary_check(payload, ThesisDraft, [])
        desc = "Stripped all citations from claims before the critic."

    elif fault == "schema_drift":
        payload = _valid_thesis_payload()
        payload["clams"] = payload.pop("claims")  # rename required field
        caught_by, errors = _boundary_check(payload, ThesisDraft, [])
        desc = "Renamed required field 'claims' -> 'clams' (schema drift)."

    elif fault == "ambiguous_ticker":
        payload = _valid_screen_payload()
        payload["candidates"][0]["ticker"] = "CATERPILLAR"  # out-of-universe alias
        caught_by, errors = _boundary_check(payload, ScreenResult, [validate_tickers_in_universe])
        desc = "Screener candidate ticker set to an out-of-universe alias."

    elif fault == "tool_truncation":
        # Force section truncation to 25% of the default limit and confirm it is marked
        # explicitly (never silent) with a [TRUNCATED] marker on the returned section.
        from desk.data import sections

        big_html = (
            "<html><body><p>Item 7. MD&amp;A</p><p>" + ("word " * 8000) + "</p></body></html>"
        )
        section_set = sections.extract_sections(
            big_html, "ACC-INJ", "10-K", max_chars_per_item=DEFAULT_MAX_SECTION_CHARS // 4
        )
        truncated = any(s.truncated for s in section_set.sections.values())
        caught_by = "tool" if truncated else "none"
        errors = ["section truncated and marked [TRUNCATED]"] if truncated else []
        desc = "Forced get_section truncation to 25% of the default limit."

    else:
        raise ValueError(f"Unknown fault: {fault!r}. Known: {FAULTS}")

    # Tag the ledger so injection runs are queryable for the failure taxonomy.
    db.record_tool_call(
        run_id=run_id,
        stage="inject",
        tool=f"fault:{fault}",
        injected_fault=fault,
        result_chars=len(str(errors)),
    )
    return InjectionResult(fault=fault, description=desc, caught_by=caught_by, errors=errors)


def run_all(run_id: str = "INJECT") -> list[InjectionResult]:
    return [run_injection(f, run_id=run_id) for f in FAULTS]
