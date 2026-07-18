"""Shared helpers for the agent-facing MCP tools.

The tool *logic* lives as pure functions in ``screen_tools`` / ``filing_tools`` so it is
unit-testable without the SDK. The ``@tool`` wrappers call those functions and format results.

A per-stage :class:`ToolContext` (a contextvar the runner sets before each stage) carries the
truncation limit and the ``run_id`` / ``stage`` / ``ticker`` used for ledger logging, so tools
stay stateless.
"""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

DEFAULT_MAX_SECTION_CHARS = 12_000
TRUNCATION_MARKER = "\n\n[TRUNCATED]"


@dataclass
class ToolContext:
    run_id: str = ""
    stage: str = ""
    ticker: str | None = None
    max_section_chars: int = DEFAULT_MAX_SECTION_CHARS
    injected_fault: str | None = None


# Default is None (a fresh ToolContext per get() call) — ContextVar defaults must be immutable.
_ctx: contextvars.ContextVar[ToolContext | None] = contextvars.ContextVar(
    "desk_tool_context", default=None
)


def set_context(ctx: ToolContext) -> contextvars.Token:
    return _ctx.set(ctx)


def reset_context(token: contextvars.Token) -> None:
    _ctx.reset(token)


def get_context() -> ToolContext:
    return _ctx.get() or ToolContext()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def truncate(text: str, max_chars: int) -> tuple[str, bool]:
    """Server-side truncation with an explicit marker (never silent)."""
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + TRUNCATION_MARKER, True
    return text, False


def tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a payload as an MCP tool result. Always stamps origin + retrieval timestamp."""
    payload.setdefault("retrieved_at", now_iso())
    return {"content": [{"type": "text", "text": json.dumps(payload, default=str)}]}


def error_result(message: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps({"error": message})}],
        "is_error": True,
    }
