"""Typer CLI entrypoint for the research desk."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="desk",
    help="Multi-agent stock research desk (engineering demo — not investment advice).",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Multi-agent stock research desk.

    Engineering demonstration of Claude Agent SDK orchestration.
    Nothing produced here is investment advice.
    """


@app.command()
def version() -> None:
    """Print the installed package version."""
    from importlib.metadata import version as _v

    try:
        typer.echo(_v("research-desk"))
    except Exception:
        typer.echo("0.1.0 (unreleased)")


@app.command()
def memo(
    query: str = typer.Argument(..., help="Natural-language screening request."),
    engine: str = typer.Option("pipeline", "--engine", help="pipeline | baseline"),
    handoff: str = typer.Option("contract", "--handoff", help="contract | full_context"),
    max_candidates: int = typer.Option(4, "--max-candidates", "-n"),
) -> None:
    """Generate a research memo (Markdown + JSON) for a screening request."""
    import asyncio

    from rich.console import Console

    from desk.orchestrator.pipeline import run_memo

    console = Console()
    result = asyncio.run(
        run_memo(query, engine=engine, max_candidates=max_candidates, handoff_mode=handoff)
    )

    console.print(f"[bold]Run[/bold] {result.run_id}  ([cyan]{engine}[/cyan])")
    for path in result.memo_paths:
        console.print(f"  memo: {path}")
    for memo_obj in result.memos:
        c = memo_obj.cost
        console.print(
            f"  [green]{memo_obj.ticker}[/green] — ${c.total_cost_usd:.4f} across "
            f"{c.n_calls} call(s), confidence {memo_obj.confidence}"
        )
    if result.failures:
        for f in result.failures:
            console.print(f"  [red]failure[/red] at {f.get('stage')}: {f.get('errors')}")
    if not result.memos:
        console.print("[red]No memo produced.[/red]")
        raise typer.Exit(code=1)


@app.command()
def costs(
    run_id: str = typer.Argument(None, help="Run id to report; omit with --all."),
    all_runs: bool = typer.Option(False, "--all", help="Show a trend table across runs."),
) -> None:
    """Show a per-stage cost breakdown for a run, or a trend table across all runs."""
    from rich.console import Console
    from rich.table import Table

    from desk.ledger import report

    console = Console()

    if all_runs:
        table = Table(title="Cost trend across runs")
        for col in ("run_id", "calls", "in tok", "out tok", "cost $", "cache save $"):
            table.add_column(col)
        for roll in report.all_runs():
            table.add_row(
                roll.run_id[:12],
                str(roll.n_llm_calls),
                f"{roll.total_input_tokens:,}",
                f"{roll.total_output_tokens:,}",
                f"{roll.total_cost_usd:.4f}",
                f"{roll.cache_savings_usd:.4f}",
            )
        console.print(table)
        return

    if not run_id:
        console.print("[red]Provide a run_id or use --all.[/red]")
        raise typer.Exit(code=1)

    roll = report.run_rollup(run_id)
    table = Table(title=f"Cost breakdown — {run_id}")
    for col in ("stage", "calls", "in tok", "out tok", "cache rd", "cost $", "degradations"):
        table.add_column(col)
    for s in roll.stages:
        table.add_row(
            s.stage,
            str(s.n_calls),
            f"{s.input_tokens:,}",
            f"{s.output_tokens:,}",
            f"{s.cache_read_tokens:,}",
            f"{s.computed_cost_usd:.4f}",
            ", ".join(sorted(set(s.degradations))) or "—",
        )
    console.print(table)
    console.print(
        f"[bold]Total[/bold]: ${roll.total_cost_usd:.4f}  ·  {roll.n_llm_calls} LLM call(s)  ·  "
        f"{roll.n_tool_calls} tool call(s) ({roll.n_truncated_tool_calls} truncated)  ·  "
        f"cache savings ${roll.cache_savings_usd:.4f}"
    )


@app.command()
def show(run_id: str = typer.Argument(..., help="Run id to display.")) -> None:
    """Pretty-print the memo(s) for a run."""
    from rich.console import Console
    from rich.markdown import Markdown

    from desk.settings import get_settings

    console = Console()
    memos_dir = get_settings().runs_dir / run_id / "memos"
    md_files = sorted(memos_dir.glob("*.md")) if memos_dir.exists() else []
    if not md_files:
        console.print(f"[red]No memos found for run {run_id}.[/red]")
        raise typer.Exit(code=1)
    for path in md_files:
        console.print(Markdown(path.read_text("utf-8")))
        console.print()


eval_app = typer.Typer(help="Evaluation harness: judge, critic, inject, compare.")
app.add_typer(eval_app, name="eval")


@eval_app.command("judge")
def eval_judge(run_id: str = typer.Argument(..., help="Run whose memos to score.")) -> None:
    """LLM-as-judge + programmatic citation accuracy for every memo in a run."""
    import asyncio

    from rich.console import Console
    from rich.table import Table

    from desk.agents.base import SdkAgentRunner
    from desk.contracts.v1 import ResearchMemo
    from desk.evals.judge import judge_memo
    from desk.settings import get_settings

    console = Console()
    memos_dir = get_settings().runs_dir / run_id / "memos"
    memo_files = sorted(memos_dir.glob("*.json")) if memos_dir.exists() else []
    if not memo_files:
        console.print(f"[red]No memos for run {run_id}.[/red]")
        raise typer.Exit(code=1)

    runner = SdkAgentRunner()

    async def _run():
        out = []
        for f in memo_files:
            memo = ResearchMemo.model_validate_json(f.read_text("utf-8"))
            out.append(await judge_memo(memo, run_id=run_id, runner=runner))
        return out

    scores = asyncio.run(_run())
    table = Table(title=f"Judge scores — {run_id}")
    for col in ("ticker", "cite acc", "ground", "balance", "specific", "readable"):
        table.add_column(col)
    for s in scores:
        table.add_row(
            s.ticker,
            f"{s.citation_accuracy:.0%}",
            str(s.grounding),
            str(s.argument_balance),
            str(s.specificity),
            str(s.readability),
        )
    console.print(table)


@eval_app.command("critic")
def eval_critic(
    n: int = typer.Option(20, "--n", help="Number of ThesisDrafts to corrupt."),
    timeout: float = typer.Option(
        180.0,
        "--timeout",
        help="Per critic-call timeout (s). Timeouts are reported separately, "
        "not counted as misses.",
    ),
) -> None:
    """Planted-error catch rate by type + isolation ON/OFF sycophancy delta."""
    import asyncio

    from rich.console import Console
    from rich.table import Table

    from desk.agents.base import SdkAgentRunner
    from desk.evals.planted_errors import evaluate_isolation_delta, load_thesis_drafts

    console = Console()
    drafts = load_thesis_drafts(limit=n)
    if not drafts:
        console.print("[red]No ThesisDrafts found in runs/. Run `desk memo` first.[/red]")
        raise typer.Exit(code=1)
    console.print(
        f"Evaluating critic on {len(drafts)} draft(s) × 4 corruption types × 2 modes "
        f"(timeout {timeout:.0f}s/call)..."
    )

    delta = asyncio.run(
        evaluate_isolation_delta(
            drafts,
            run_id="critic-eval",
            runner=SdkAgentRunner(),
            timeout=timeout,
            progress=lambda msg: console.print(f"  [dim]{msg}[/dim]"),
        )
    )
    on, off = delta.isolation_on, delta.isolation_off
    table = Table(title="Critic catch rate by corruption type (over completed calls)")
    for col in ("corruption", "ON catch", "ON timeouts", "OFF catch", "OFF timeouts"):
        table.add_column(col)

    def _catch(res, kind: str) -> str:
        completed = res.by_type_completed.get(kind, 0)
        return f"{res.by_type_catch[kind]:.0%} (n={completed})" if completed else "— (n=0)"

    for kind in on.by_type_catch:
        table.add_row(
            kind,
            _catch(on, kind),
            str(on.by_type_timeout.get(kind, 0)),
            _catch(off, kind),
            str(off.by_type_timeout.get(kind, 0)),
        )
    console.print(table)
    console.print(
        f"[bold]Overall catch (completed only):[/bold] ON {on.overall_catch_rate:.0%} vs "
        f"OFF {off.overall_catch_rate:.0%}  →  sycophancy delta {delta.catch_rate_delta:+.0%}"
    )
    if on.n_timeouts or off.n_timeouts:
        console.print(
            f"[yellow]Timeouts:[/yellow] ON {on.n_timeouts} / OFF {off.n_timeouts} calls exceeded "
            f"{timeout:.0f}s (excluded from catch rate). Raise --timeout for cleaner citation-"
            f"corruption numbers."
        )
    console.print(
        f"False-alarm rate (proxy): ON {on.false_alarm_rate:.2f} / "
        f"OFF {off.false_alarm_rate:.2f} material+ challenges per claim"
    )


@eval_app.command("inject")
def eval_inject(
    fault: str = typer.Option(None, "--fault", help="Fault name; omit to run all."),
) -> None:
    """Handoff failure-injection experiments; reports what boundary validation catches."""
    from rich.console import Console
    from rich.table import Table

    from desk.evals.failure_injection import FAULTS, run_all, run_injection

    console = Console()
    if fault and fault not in FAULTS:
        console.print(f"[red]Unknown fault {fault!r}. Known: {FAULTS}[/red]")
        raise typer.Exit(code=1)
    results = [run_injection(fault)] if fault else run_all()
    table = Table(title="Failure injection")
    for col in ("fault", "caught by", "description"):
        table.add_column(col)
    for r in results:
        table.add_row(r.fault, r.caught_by, r.description)
    console.print(table)


@eval_app.command("compare")
def eval_compare(
    queries: int = typer.Option(12, "--queries", help="Number of committed queries to run."),
    seeds: int = typer.Option(1, "--seeds", help="Seeds per (query, engine)."),
    max_candidates: int = typer.Option(3, "--max-candidates", "-n"),
    concurrency: int = typer.Option(
        1, "--concurrency", help="Full screens to run at once (rate-limit guard)."
    ),
    out: str = typer.Option(None, "--out", help="Write the JSON export to this path."),
    eval_id: str = typer.Option(
        None,
        "--eval-id",
        help="Stable id for a resumable run. Re-invoke with the same id to finish only the "
        "screens that didn't complete (kill-safe). Omit to start a fresh run.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of Markdown."),
) -> None:
    """Baseline vs pipeline comparison table on the committed queries."""
    import asyncio
    import json as _json
    from pathlib import Path

    from rich.console import Console

    from desk.evals.compare import COMMITTED_QUERIES, render_markdown, run_comparison
    from desk.orchestrator.run import new_run_id
    from desk.settings import get_settings

    console = Console()
    qs = COMMITTED_QUERIES[: max(1, queries)]
    eval_id = eval_id or new_run_id()
    eval_dir = get_settings().runs_dir / "evals" / "compare" / eval_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Compare eval [cyan]{eval_id}[/cyan] · {len(qs)} query(ies) · resumable")

    cmp = asyncio.run(
        run_comparison(
            queries=qs,
            seeds=seeds,
            max_candidates=max_candidates,
            concurrency=concurrency,
            eval_dir=eval_dir,
            progress=lambda msg: console.print(f"  [dim]{msg}[/dim]"),
        )
    )
    # Always persist the assembled export alongside the per-screen unit files.
    (eval_dir / "compare_export.json").write_text(_json.dumps(cmp.as_dict(), indent=2), "utf-8")
    if out:
        Path(out).write_text(_json.dumps(cmp.as_dict(), indent=2), "utf-8")
        console.print(f"[green]Wrote[/green] {out}")
    if as_json:
        console.print_json(_json.dumps(cmp.as_dict()))
    else:
        console.print(render_markdown(cmp))


@eval_app.command("screener")
def eval_screener(
    models: str = typer.Option(
        "default", "--models", help="Comma-separated: default,haiku,sonnet,opus (or full ids)."
    ),
    query: str = typer.Option(None, "--query", help="Run only this suite query id."),
    repeats: int = typer.Option(2, "--repeats", help="Seeds per (query, variant)."),
    concurrency: int = typer.Option(
        4, "--concurrency", help="Max screener calls in flight at once (rate-limit guard)."
    ),
    report_id: str = typer.Option(
        None, "--report", help="Re-render a finished eval run from disk; do not re-run agents."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the results JSON instead of the report."),
) -> None:
    """Stage-level, deterministic screener eval against the synthetic fixture universe.

    Runs the real screener over a model matrix, scores its intent->filters translation
    (golden / paraphrase / perturbation / relaxation / coverage), and writes a report. Exits
    non-zero only on harness errors — a low score is a finding, not a failure.
    """
    import asyncio
    import json as _json

    from rich.console import Console

    from desk.evals.screener import harness, report
    from desk.evals.screener.suite import load_suite
    from desk.settings import get_settings

    console = Console()
    eval_dir_root = get_settings().runs_dir / "evals" / "screener"

    # --- re-render mode: no agents, just redraw a saved run ---
    if report_id:
        results_path = eval_dir_root / report_id / "results.json"
        if not results_path.exists():
            console.print(f"[red]No saved eval at {results_path}.[/red]")
            raise typer.Exit(code=1)
        saved = _json.loads(results_path.read_text("utf-8"))
        evals = report.from_json(saved["results"])
        md = report.render_report(evals, eval_run_id=report_id, repeats=saved.get("repeats", 1))
        (eval_dir_root / report_id / "report.md").write_text(md, "utf-8")
        console.print(md if not as_json else _json.dumps(saved, indent=2))
        return

    # --- run mode: drive the real screener (live model calls) ---
    from desk.agents.base import SdkAgentRunner
    from desk.orchestrator.run import new_run_id

    suite = load_suite()
    if query:
        suite = [q for q in suite if q.id == query]
        if not suite:
            console.print(f"[red]No suite query with id {query!r}.[/red]")
            raise typer.Exit(code=1)

    model_list = harness.resolve_models(models)
    eval_run_id = new_run_id()
    eval_dir = eval_dir_root / eval_run_id
    eval_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"Screener eval [cyan]{eval_run_id}[/cyan] · models {model_list} · "
        f"{len(suite)} query(ies) · repeats {repeats} · concurrency {concurrency}"
    )

    try:
        evals = asyncio.run(
            harness.run_matrix(
                model_list,
                suite,
                runner=SdkAgentRunner(),
                run_id=eval_run_id,
                repeats=repeats,
                concurrency=concurrency,
                progress=lambda msg: console.print(f"  [dim]{msg}[/dim]"),
            )
        )
    except Exception as exc:  # noqa: BLE001 — harness error, distinct from a low score
        console.print(f"[red]Harness error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    payload = {"eval_run_id": eval_run_id, "repeats": repeats, "results": report.to_json(evals)}
    (eval_dir / "results.json").write_text(_json.dumps(payload, indent=2), "utf-8")
    md = report.render_report(evals, eval_run_id=eval_run_id, repeats=repeats)
    (eval_dir / "report.md").write_text(md, "utf-8")

    if as_json:
        console.print_json(_json.dumps(payload))
    else:
        console.print(md)
    console.print(f"[green]Wrote[/green] {eval_dir}/report.md")


universe_app = typer.Typer(help="Manage the screening universe and its metrics table.")
app.add_typer(universe_app, name="universe")


@universe_app.command("build")
def universe_build(
    force: bool = typer.Option(False, "--force", help="Re-fetch and recompute, ignoring cache."),
) -> None:
    """Build/refresh the derived-metrics table for the default universe."""
    from rich.console import Console
    from rich.table import Table

    from desk.data import universe

    console = Console()
    tickers = universe.universe_tickers()
    console.print(
        f"Building metrics for {len(tickers)} tickers ({'forced' if force else 'cache-first'})..."
    )
    rows = universe.build(force=force)

    table = Table(title="Universe metrics")
    for col in ("ticker", "sector", "mkt cap ($B)", "P/E", "rev growth", "op margin", "FY"):
        table.add_column(col)
    for m in rows:
        table.add_row(
            m.ticker,
            m.sector or "—",
            f"{m.market_cap / 1e9:.1f}" if m.market_cap else "—",
            f"{m.trailing_pe:.1f}" if m.trailing_pe else "—",
            f"{m.revenue_growth_yoy:+.1%}" if m.revenue_growth_yoy is not None else "—",
            f"{m.operating_margin:.1%}" if m.operating_margin is not None else "—",
            str(m.fiscal_year or "—"),
        )
    console.print(table)
    console.print(f"[green]Built {len(rows)} metric rows.[/green]")


if __name__ == "__main__":
    app()
