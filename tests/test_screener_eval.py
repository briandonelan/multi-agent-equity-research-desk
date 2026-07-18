"""M7.1 acceptance: the fixture universe, the suite, and their golden labels are self-consistent
*before* any LLM is involved. Also exercises the MetricsSource injection."""

from __future__ import annotations

from desk.data import universe
from desk.evals.screener.fixtures import fixture_tickers, load_fixture_source
from desk.evals.screener.suite import load_dimension_map, load_suite
from desk.tools import screen_tools


def test_fixture_loads_with_expected_shape():
    src = load_fixture_source()
    rows = src.get_table()
    assert len(rows) >= 30
    sectors = {r.sector for r in rows}
    # Three deep sectors + a long tail.
    for deep in ("Industrials", "Technology", "Healthcare"):
        assert sum(1 for r in rows if r.sector == deep) >= 8
    assert {"Energy", "Financials", "Materials", "Utilities"} <= sectors
    # Boundary + None cases are present and don't crash the source.
    assert src.get_row("IND8").market_cap == 10_000_000_000  # the mid/large boundary
    assert src.get_row("IND1").trailing_pe is None  # null P/E
    assert src.get_row("FIN1").operating_margin is None  # null margins


def test_golden_filters_reproduce_expected_tickers():
    """Every query's hand-written golden_filters must reproduce its expected_tickers exactly."""
    rows = load_fixture_source().get_table()
    for q in load_suite():
        got = {r["ticker"] for r in universe.run_screen(q.golden_filters, limit=10, rows=rows)}
        assert got == set(q.expected_tickers), (
            f"{q.id}: golden_filters returned {sorted(got)}, expected {sorted(q.expected_tickers)}"
        )


def test_starve_queries_actually_starve():
    rows = load_fixture_source().get_table()
    for q in load_suite():
        if q.starve:
            got = universe.run_screen(q.golden_filters, limit=10, rows=rows)
            assert len(got) < 2, f"{q.id} was supposed to starve but matched {len(got)}"
            assert q.expected_relaxation_field is not None


def test_midcap_boundary_excludes_exactly_10b():
    """IND8 sits at exactly $10B; the upper-exclusive mid-cap filter must drop it."""
    rows = load_fixture_source().get_table()
    mid = {r["ticker"] for r in universe.run_screen(
        [{"field": "market_cap", "op": ">=", "value": 2_000_000_000},
         {"field": "market_cap", "op": "<", "value": 10_000_000_000}],
        limit=10, rows=rows)}
    assert "IND8" not in mid
    assert "IND2" in mid


def test_dimension_map_covers_all_suite_dimensions():
    dim_map = load_dimension_map()
    for q in load_suite():
        for dim in q.dimensions:
            assert dim in dim_map, f"{q.id}: dimension {dim!r} missing from dimension_map.yaml"


def test_metrics_source_injection_into_tools():
    """The screen tools read whatever source they're given, not the global universe."""
    src = load_fixture_source()
    out = screen_tools.get_metrics_logic("IND7", source=src)
    assert out["metrics"]["ticker"] == "IND7"
    screened = screen_tools.run_screen_logic(
        [{"field": "sector", "op": "==", "value": "Industrials"}], limit=10, source=src
    )
    assert screened["count"] >= 8
    assert all(r["sector"] == "Industrials" for r in screened["rows"])


def test_fixture_tickers_helper():
    tickers = fixture_tickers()
    assert "IND1" in tickers and "TEC1" in tickers and "HLT1" in tickers


# --- M7.2: contract v1.1 (structured relaxation disclosure) ---


def _v1_fixture_path():
    from pathlib import Path

    return Path(__file__).parent / "fixtures" / "screener" / "screen_result_v1.json"


def test_v1_screenresult_loads_under_v11_backward_compatible():
    """An old v1 artifact (no `relaxations` key) loads cleanly, defaulting relaxations to []."""
    from desk.contracts.v1 import ScreenResult

    m = ScreenResult.model_validate_json(_v1_fixture_path().read_text("utf-8"))
    assert m.schema_version == "1"  # provenance preserved from the old artifact
    assert m.relaxations == []
    assert [c.ticker for c in m.candidates] == ["IND2", "IND7"]


def test_registry_maps_v1_and_v11_to_same_models():
    from desk.contracts import registry, v1

    assert registry.models_for("1") is v1
    assert registry.models_for("1.1") is v1


def test_relaxation_roundtrip_v11():
    from desk.contracts.v1 import Relaxation, ScreenFilter, ScreenResult

    r = Relaxation(
        field="net_debt_to_ebitda",
        original=ScreenFilter(field="net_debt_to_ebitda", op="<", value=0.0),
        relaxed_to=ScreenFilter(field="net_debt_to_ebitda", op="<", value=1.0),
        reason="No net-cash names matched; loosened to low leverage.",
    )
    result = ScreenResult(
        run_id="r",
        interpreted_intent="i (relaxed net debt)",
        filters_applied=[],
        relaxations=[r],
        candidates=[{"ticker": "HLT6", "cik": "1", "company_name": "Larkspur Labs", "rationale": "x"}],
    )
    assert result.schema_version == "1.1"
    reloaded = ScreenResult.model_validate_json(result.model_dump_json())
    assert reloaded.relaxations[0].field == "net_debt_to_ebitda"
    assert reloaded.relaxations[0].relaxed_to.value == 1.0
