"""Render a screener-eval run to Markdown, and serialize/reload it for re-rendering.

The JSON form lets ``desk eval screener --report <id>`` redraw the Markdown without re-running the
(expensive) agent. The report has three parts: a per-check summary table across models, a
per-query drill-down ordered worst-first, and an ``<!-- ANALYSIS -->`` placeholder for a written
takeaway.
"""

from __future__ import annotations

from dataclasses import asdict

from desk.evals.screener.scoring import (
    ModelEvaluation,
    PerturbSummary,
    QueryEvaluation,
    RelaxationSummary,
)


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


# --- serialization ---------------------------------------------------------------------------


def to_json(evaluations: list[ModelEvaluation]) -> dict:
    return {
        "schema": "screener-eval/1",
        "models": [
            {"model": ev.model, "queries": [asdict(q) for q in ev.queries]}
            for ev in evaluations
        ],
    }


def from_json(data: dict) -> list[ModelEvaluation]:
    out: list[ModelEvaluation] = []
    for m in data.get("models", []):
        queries: list[QueryEvaluation] = []
        for q in m.get("queries", []):
            q = dict(q)
            perts = [PerturbSummary(**p) for p in q.pop("perturbations", [])]
            rel = q.pop("relaxation", None)
            qe = QueryEvaluation(**q)
            qe.perturbations = perts
            qe.relaxation = RelaxationSummary(**rel) if rel else None
            queries.append(qe)
        out.append(ModelEvaluation(model=m["model"], queries=queries))
    return out


# --- rendering -------------------------------------------------------------------------------


def _summary_table(evaluations: list[ModelEvaluation]) -> str:
    rows = [
        ("Golden recall", "mean_recall"),
        ("Golden precision", "mean_precision"),
        ("Dimension coverage", "mean_coverage"),
        ("Paraphrase stability", "mean_paraphrase"),
        ("Perturbation targeted-move", "mean_targeted_move"),
        ("Perturbation off-target", "mean_off_target"),
        ("Relaxation disclosed", "relaxation_disclosed_rate"),
    ]
    header = "| Check | " + " | ".join(ev.model for ev in evaluations) + " |"
    sep = "| --- | " + " | ".join("---" for _ in evaluations) + " |"
    lines = [header, sep]
    for label, attr in rows:
        cells = [_pct(getattr(ev, attr)) for ev in evaluations]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    # Errors row is a count, not a percentage.
    err_cells = [str(ev.n_errors) for ev in evaluations]
    lines.append("| Handoff errors | " + " | ".join(err_cells) + " |")
    return "\n".join(lines)


def _query_headline(q: QueryEvaluation) -> float:
    """Lowest applicable rate — used to sort the drill-down worst-first."""
    candidates = [
        q.golden_recall,
        q.golden_precision,
        q.coverage_rate,
        q.paraphrase_jaccard,
    ]
    if q.relaxation is not None:
        candidates.append(q.relaxation.fully_disclosed_rate)
    for p in q.perturbations:
        candidates.append(p.targeted_move_rate)
        candidates.append(1.0 - p.off_target_change_rate)
    vals = [c for c in candidates if c is not None]
    return min(vals) if vals else 1.0


def _query_block(q: QueryEvaluation) -> str:
    lines = [f"### {q.query_id}  (worst rate {_query_headline(q):.0%})"]
    if q.n_errors:
        lines.append(f"- **handoff errors:** {q.n_errors}/{q.n_seeds} seeds")
    if q.starve:
        r: RelaxationSummary | None = q.relaxation
        if r:
            lines.append(
                f"- **relaxation:** correct field `{r.expected_field}` "
                f"{_pct(r.correct_field_rate)} · structured {_pct(r.structured_rate)} · "
                f"prose {_pct(r.prose_rate)} · fully disclosed {_pct(r.fully_disclosed_rate)}"
            )
    else:
        miss = ", ".join(q.golden_missing) or "none"
        extra = ", ".join(q.golden_unexpected) or "none"
        lines.append(
            f"- **golden:** recall {_pct(q.golden_recall)} · "
            f"precision {_pct(q.golden_precision)} · missing: {miss} · unexpected: {extra}"
        )
    unc = ", ".join(q.coverage_uncovered) or "none"
    lines.append(f"- **coverage:** {_pct(q.coverage_rate)} · uncovered dims: {unc}")
    if q.paraphrase_jaccard is not None:
        lines.append(f"- **paraphrase stability:** {_pct(q.paraphrase_jaccard)} mean Jaccard")
    for p in q.perturbations:
        lines.append(
            f"- **perturbation** _{p.text}_: targeted-move {_pct(p.targeted_move_rate)} · "
            f"off-target {_pct(p.off_target_change_rate)}"
        )
    return "\n".join(lines)


def render_report(
    evaluations: list[ModelEvaluation], *, eval_run_id: str, repeats: int
) -> str:
    parts = [
        f"# Screener evaluation — `{eval_run_id}`",
        "",
        "Stage-level, deterministic evaluation of the screener's natural-language → filters "
        "translation against a synthetic fixture universe. No LLM judge: the filters run "
        "mechanically, so every number below is exact.",
        "",
        f"Models: {', '.join(ev.model for ev in evaluations)} · repeats: {repeats} · "
        f"queries: {len(evaluations[0].queries) if evaluations else 0}",
        "",
        "## Summary",
        "",
        _summary_table(evaluations),
        "",
        "## Per-query drill-down (worst first)",
        "",
    ]
    # Drill-down uses the first model's ordering; that is the one under primary scrutiny.
    primary = evaluations[0] if evaluations else None
    if primary:
        ordered = sorted(primary.queries, key=_query_headline)
        parts.extend(_query_block(q) + "\n" for q in ordered)
    parts.extend(["", "<!-- ANALYSIS -->", ""])
    return "\n".join(parts)
