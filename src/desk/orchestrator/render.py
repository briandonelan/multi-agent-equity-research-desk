"""Render a :class:`ResearchMemo` to Markdown."""

from __future__ import annotations

from desk.contracts.v1 import ResearchMemo


def _fmt_val(key: str, val: float | None) -> str:
    if val is None:
        return "n/a"
    if "margin" in key or "growth" in key:
        return f"{val:.1%}"
    if "market_cap" in key or "revenue" in key:
        return f"${val / 1e9:.1f}B"
    if "pe" in key:
        return f"{val:.1f}x"
    return f"{val:,.2f}"


def render_memo(memo: ResearchMemo) -> str:
    lines: list[str] = []
    lines.append(f"# {memo.company_name} ({memo.ticker}) — Research Memo")
    lines.append("")
    lines.append(f"*Run `{memo.run_id}` · as of {memo.as_of} · confidence: **{memo.confidence}***")
    lines.append("")
    lines.append("## Thesis")
    lines.append(memo.thesis_summary)
    lines.append("")

    if memo.valuation_snapshot:
        lines.append("## Valuation snapshot")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        for k, v in memo.valuation_snapshot.items():
            lines.append(f"| {k} | {_fmt_val(k, v)} |")
        lines.append("")

    lines.append("## Bull case")
    lines.append("")
    if memo.bull_case:
        for i, claim in enumerate(memo.bull_case, 1):
            lines.append(f"{i}. **[{claim.claim_type}]** {claim.text}")
            for cite in claim.citations:
                lines.append(f"   - > {cite.quote}  \n     `{cite.source_ref}`")
    else:
        lines.append("_No bull-case claims survived synthesis._")
    lines.append("")

    lines.append("## Bear case")
    lines.append("")
    if memo.bear_case:
        for i, ch in enumerate(memo.bear_case, 1):
            target = (
                "" if ch.target_claim_idx is None else f" (re: claim {ch.target_claim_idx + 1})"
            )
            lines.append(f"{i}. **[{ch.severity}]**{target} {ch.text}")
            for cite in ch.citations:
                lines.append(f"   - > {cite.quote}  \n     `{cite.source_ref}`")
    else:
        lines.append(
            "_No bear-case challenges were recorded (critique may have been "
            "unavailable — see unresolved disagreements)._"
        )
    lines.append("")

    if memo.unresolved_disagreements:
        lines.append("## Unresolved disagreements")
        lines.append("")
        for d in memo.unresolved_disagreements:
            lines.append(f"- {d}")
        lines.append("")

    lines.append("## Confidence")
    lines.append(f"**{memo.confidence}** — {memo.confidence_rationale}")
    lines.append("")

    c = memo.cost
    lines.append("## Cost")
    lines.append(
        f"This memo cost **${c.total_cost_usd:.4f}** across {c.n_calls} model call(s) "
        f"({c.total_input_tokens:,} in / {c.total_output_tokens:,} out tokens"
        + (f", {c.total_cache_read_tokens:,} cache-read" if c.total_cache_read_tokens else "")
        + ")."
    )
    if c.by_stage_cost_usd:
        lines.append("")
        for stage, cost in c.by_stage_cost_usd.items():
            lines.append(f"- {stage}: ${cost:.4f}")
    lines.append("")

    lines.append("---")
    lines.append(f"_{memo.disclaimer}_")
    lines.append("")
    return "\n".join(lines)
