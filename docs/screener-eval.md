# Screener evaluation harness — design note

The screener is the first stage of the desk: it turns a natural-language request
(*"undervalued mid-cap industrials with improving margins"*) into declarative filters and runs
them against a metrics universe. Everything downstream inherits its mistakes, so it is the stage
most worth evaluating in isolation — and, conveniently, the one that can be evaluated *without an
LLM judge*.

## Why this stage can be scored deterministically

`run_screen` executes filters mechanically: given filters and a table, the matching rows are a
pure function of the two. So once a request is translated into filters, there is no further
judgement — the shortlist is exact. The *only* thing that can be wrong is the
**intent → filters translation**. That means we can:

1. Write a synthetic universe where we control every metric.
2. Hand-write the canonical ("golden") filters for a request and confirm they reproduce a labelled
   set of tickers — with no model in the loop (`test_golden_filters_reproduce_expected_tickers`).
3. Run the real screener on the request and compare *its* shortlist to the golden one.

No rubric, no grader model, no ceiling effects. A wrong number is an exact, reproducible defect.

## Operationalizing the fuzzy terms (the judgement calls)

A request like "undervalued" has no canonical threshold. Rather than hide that behind a model, we
fixed each fuzzy term to an explicit rule and documented it in the fixture header. These are
**choices**, defensible but not unique; the point is that they are written down and testable:

| Phrase | Operationalized as |
| --- | --- |
| mid-cap | `market_cap` in `[2e9, 10e9)` — **upper-exclusive** (a name at exactly $10B is large-cap) |
| undervalued | `trailing_pe < 15` |
| cheap | `trailing_pe < 12` |
| improving margins | `operating_margin_trend > 0` (or `gross_margin_trend > 0`) |
| profitable | `operating_margin > 0` |
| quality | `operating_margin > 0.25` and `net_debt_to_ebitda < 1` |
| low leverage | `net_debt_to_ebitda < 1` |
| net cash | `net_debt_to_ebitda < 0` |
| strong growth | `revenue_growth_yoy > 0.15` |
| steady revenue | `revenue_growth_yoy` in `[0, 0.05]` |

The mid-cap boundary is deliberately a *test*: the fixture has a ticker sitting at exactly
`market_cap = 10.0e9`, and `test_midcap_boundary_excludes_exactly_10b` pins that the canonical
filter drops it. It documents the convention and guards against an off-by-one `<=` creeping in.

## The fixture universe

36 synthetic tickers: three deep sectors (Industrials, Technology, Healthcare, ≥9 each) plus a
long tail (Energy, Financials, Materials, Utilities). It is built to *stress translation*, not to
be realistic:

- **Boundary cases** — a name at exactly the mid/large-cap line.
- **Nulls** — loss-makers with no clean `trailing_pe`; financials with no margin fields. A correct
  translation must not silently match a null.
- **The trap** — a name that is cheap on P/E but has collapsing margins, to catch a screener that
  keys on "cheap" and ignores "improving".
- **Near-duplicates** — two names that differ on a single metric, so a dropped filter changes the
  answer.

Synthetic data also sidesteps look-ahead bias and keeps the eval fully offline (no live prices),
which is what lets the whole harness run under a network-blocking test.

## The five checks

Each is a pure function over the screener's `ScreenResult`(s); none uses a model.

- **Golden** — recall and precision of the shortlist against the labelled expected set.
  `acceptable_extra` tickers are defensible either-way inclusions and never count against
  precision.
- **Paraphrase** — the base request plus five rewordings should mean the same thing; we report the
  mean pairwise Jaccard overlap of their shortlists. Low overlap means the translation is
  phrasing-sensitive.
- **Perturbation** — a one-dimension edit to the request (e.g. "improving" → "deteriorating"
  margins) should move the filter on the affected field and leave the others alone. Reported as a
  targeted-move rate and an off-target-change rate.
- **Relaxation** — for *starve* queries (written so fewer than two rows match), the screener must
  loosen the least-central filter once and disclose it twice: as a structured `relaxations` entry
  **and** in one sentence of prose. Reported, never hard-failed — "which filter is least central"
  is itself a judgement call.
- **Coverage** — every dimension the request is annotated with must produce at least one filter on
  one of that dimension's metric fields (via `dimension_map.yaml`). A missed dimension means part
  of the request was ignored.

## Contract evolution

Structured relaxation disclosure was added as schema **v1.1**: an additive, backward-compatible
`relaxations` list on `ScreenResult`. A v1 artifact with no `relaxations` key loads with an empty
list (`test_v1_screenresult_loads_under_v11_backward_compatible`), so old committed runs still
parse. The registry maps both `"1"` and `"1.1"` to the same model module.

## Running it

```
# Re-render a finished run without touching the agent (cheap, offline):
desk eval screener --report <eval_run_id>

# Live run over a model matrix (costs real tokens):
desk eval screener --models haiku,sonnet --repeats 2
desk eval screener --query q03_profitable_midcap_industrials_improving
```

Results land under `runs/evals/screener/<eval_run_id>/` as `results.json` (re-renderable) and
`report.md` (summary table per check × model, per-query drill-down worst-first, and an
`<!-- ANALYSIS -->` placeholder for the written takeaway). The command exits non-zero only on a
harness error — a low score is a finding, not a failure.

> This is an engineering demonstration. Nothing here is investment advice.
