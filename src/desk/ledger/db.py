"""SQLite token/cost ledger.

Two tables:

- ``llm_calls`` — one row per agent stage call, populated from the SDK's usage/result message.
  ``computed_cost_usd`` is ALWAYS derived from ``config/pricing.yaml`` (not the SDK's reported
  cost) so tier/pricing experiments re-price consistently.
- ``tool_calls`` — one row per MCP tool invocation, populated from PreToolUse/PostToolUse hooks.

The DB lives under the cache dir so it persists across runs (for ``desk costs --all``) while
tests get automatic isolation via the temp cache dir.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from desk.settings import get_settings, load_yaml_config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    ticker TEXT,
    model TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reported_cost_usd REAL,
    computed_cost_usd REAL DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    n_turns INTEGER DEFAULT 0,
    degradations TEXT,
    injected_fault TEXT,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    ticker TEXT,
    tool TEXT NOT NULL,
    args_chars INTEGER DEFAULT 0,
    result_chars INTEGER DEFAULT 0,
    truncated INTEGER DEFAULT 0,
    injected_fault TEXT,
    ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_run ON llm_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_tool_run ON tool_calls(run_id);
"""


def _db_path() -> Path:
    d = get_settings().cache_dir
    d.mkdir(parents=True, exist_ok=True)
    return d / "ledger.sqlite"


@contextmanager
def connect():
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# --- Pricing --------------------------------------------------------------------------------


def compute_cost(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Dollar cost from ``config/pricing.yaml`` (per-MTok). Unknown model -> 0.0."""
    prices = load_yaml_config("pricing").get("models", {}).get(model)
    if not prices:
        return 0.0
    per = 1_000_000
    return round(
        input_tokens / per * prices.get("input", 0.0)
        + output_tokens / per * prices.get("output", 0.0)
        + cache_read_tokens / per * prices.get("cache_read", 0.0)
        + cache_write_tokens / per * prices.get("cache_write", 0.0),
        6,
    )


# --- Writers --------------------------------------------------------------------------------


def record_llm_call(
    *,
    run_id: str,
    stage: str,
    model: str,
    ticker: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    reported_cost_usd: float | None = None,
    latency_ms: int = 0,
    n_turns: int = 0,
    degradations: list[str] | None = None,
    injected_fault: str | None = None,
) -> float:
    """Insert an LLM-call row; returns the computed cost so the caller can stamp the artifact."""
    computed = compute_cost(
        model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    with connect() as conn:
        conn.execute(
            """INSERT INTO llm_calls
               (run_id, stage, ticker, model, input_tokens, output_tokens,
                cache_read_tokens, cache_write_tokens, reported_cost_usd, computed_cost_usd,
                latency_ms, n_turns, degradations, injected_fault, ts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                stage,
                ticker,
                model,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_write_tokens,
                reported_cost_usd,
                computed,
                latency_ms,
                n_turns,
                json.dumps(degradations or []),
                injected_fault,
                _now_iso(),
            ),
        )
    return computed


def record_tool_call(
    *,
    run_id: str,
    stage: str,
    tool: str,
    ticker: str | None = None,
    args_chars: int = 0,
    result_chars: int = 0,
    truncated: bool = False,
    injected_fault: str | None = None,
) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT INTO tool_calls
               (run_id, stage, ticker, tool, args_chars, result_chars, truncated,
                injected_fault, ts)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                stage,
                ticker if ticker else None,
                tool,
                args_chars,
                result_chars,
                1 if truncated else 0,
                injected_fault,
                _now_iso(),
            ),
        )
