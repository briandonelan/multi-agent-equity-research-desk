"""AgentStage runner: options, usage capture, ledger, boundary validation, repair-retry.

Each stage is an independent SDK ``query()`` call with its own options. The only thing crossing
a stage boundary is a validated contract artifact. This module provides:

- :class:`AgentRunner` — the interface the stage calls. :class:`SdkAgentRunner` drives the real
  Claude Agent SDK; :class:`FakeAgentRunner` replays recorded outputs for offline tests.
- :class:`AgentStage` — builds options, runs the agent, logs the call to the ledger, then
  parses -> Pydantic-validates -> semantically-validates the output, with at most **one**
  repair retry. Two failures raise :class:`~desk.contracts.v1.HandoffFailure`.
- Semantic validators: citations resolve to cached passages and quotes fuzzy-match; tickers are
  in-universe.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from desk.contracts.v1 import Artifact, HandoffFailure, StageCost
from desk.data import sections
from desk.data import universe as universe_mod
from desk.ledger import db
from desk.tools.util import ToolContext, reset_context, set_context

# --- Runner interface -----------------------------------------------------------------------


@dataclass
class AgentResult:
    text: str
    structured_output: dict | None = None
    usage: dict = field(default_factory=dict)
    reported_cost_usd: float | None = None
    latency_ms: int = 0
    n_turns: int = 0
    model: str = ""


@dataclass
class StageSpec:
    stage: str
    model: str
    system_prompt: str
    prompt: str
    allowed_tools: list[str] = field(default_factory=list)
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    max_turns: int = 12
    output_schema: dict | None = None


class AgentRunner(Protocol):
    async def run(self, spec: StageSpec, run_id: str) -> AgentResult: ...


# --- SDK runner -----------------------------------------------------------------------------


class SdkAgentRunner:
    """Drives the real Claude Agent SDK. Captures usage/cost and logs tool calls via hooks."""

    def __init__(self, *, permission_mode: str = "bypassPermissions"):
        self.permission_mode = permission_mode

    async def run(self, spec: StageSpec, run_id: str) -> AgentResult:
        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

        hooks = _make_tool_logging_hooks(run_id, spec.stage)
        # A string system_prompt REPLACES the SDK's default system-prompt preset (verified:
        # ~18K -> ~3K tokens). `allowed_tools` whitelists only our read-only MCP tools;
        # `disallowed_tools` blocks the dangerous built-ins (Bash/Write/Edit/web). An empty
        # `setting_sources` keeps project and user settings files out of the agent context.
        options = ClaudeAgentOptions(
            system_prompt=spec.system_prompt,
            model=spec.model,
            max_turns=spec.max_turns,
            allowed_tools=spec.allowed_tools,
            disallowed_tools=[
                "Bash",
                "Write",
                "Edit",
                "NotebookEdit",
                "WebSearch",
                "WebFetch",
                "Task",
            ],
            mcp_servers=spec.mcp_servers,
            permission_mode=self.permission_mode,
            setting_sources=[],
            hooks=hooks,
        )

        start = time.monotonic()
        result_msg = None
        async for message in query(prompt=spec.prompt, options=options):
            if isinstance(message, ResultMessage):
                result_msg = message
        latency_ms = int((time.monotonic() - start) * 1000)

        if result_msg is None:
            return AgentResult(text="", latency_ms=latency_ms, model=spec.model)

        usage = _normalize_usage(getattr(result_msg, "usage", None))
        return AgentResult(
            text=getattr(result_msg, "result", "") or "",
            structured_output=getattr(result_msg, "structured_output", None),
            usage=usage,
            reported_cost_usd=getattr(result_msg, "total_cost_usd", None),
            latency_ms=latency_ms,
            n_turns=getattr(result_msg, "num_turns", 0) or 0,
            model=spec.model,
        )


def _normalize_usage(usage: Any) -> dict:
    """Extract token counts from the SDK usage object/dict defensively."""
    if usage is None:
        return {}
    d = usage if isinstance(usage, dict) else getattr(usage, "__dict__", {})
    return {
        "input_tokens": int(d.get("input_tokens", 0) or 0),
        "output_tokens": int(d.get("output_tokens", 0) or 0),
        "cache_read_tokens": int(
            d.get("cache_read_input_tokens", d.get("cache_read_tokens", 0)) or 0
        ),
        "cache_write_tokens": int(
            d.get("cache_creation_input_tokens", d.get("cache_write_tokens", 0)) or 0
        ),
    }


def _make_tool_logging_hooks(run_id: str, stage: str):
    """PreToolUse/PostToolUse hooks that log every MCP tool call to the ledger."""
    from claude_agent_sdk import HookMatcher

    pending: dict[str, int] = {}

    async def pre_tool(input_data, tool_use_id, context):  # noqa: ANN001
        try:
            name = input_data.get("tool_name", "")
            args = input_data.get("tool_input", {})
            pending[tool_use_id or name] = len(json.dumps(args, default=str))
        except Exception:  # noqa: BLE001
            pass
        return {}

    async def post_tool(input_data, tool_use_id, context):  # noqa: ANN001
        try:
            name = input_data.get("tool_name", "")
            response = input_data.get("tool_response", input_data.get("tool_result", ""))
            result_chars = len(json.dumps(response, default=str))
            truncated = "[TRUNCATED]" in json.dumps(response, default=str)
            db.record_tool_call(
                run_id=run_id,
                stage=stage,
                tool=name,
                args_chars=pending.pop(tool_use_id or name, 0),
                result_chars=result_chars,
                truncated=truncated,
            )
        except Exception:  # noqa: BLE001
            pass
        return {}

    return {
        "PreToolUse": [HookMatcher(hooks=[pre_tool])],
        "PostToolUse": [HookMatcher(hooks=[post_tool])],
    }


# --- Fake runner (offline tests) ------------------------------------------------------------


class FakeAgentRunner:
    """Replays recorded outputs. Queue one AgentResult (or raw JSON text) per expected call."""

    def __init__(self, outputs: list[AgentResult | str | dict]):
        self._queue: list[AgentResult] = []
        for o in outputs:
            if isinstance(o, AgentResult):
                self._queue.append(o)
            elif isinstance(o, dict):
                self._queue.append(AgentResult(text=json.dumps(o)))
            else:
                self._queue.append(AgentResult(text=o))
        self.calls: list[StageSpec] = []

    async def run(self, spec: StageSpec, run_id: str) -> AgentResult:
        self.calls.append(spec)
        if not self._queue:
            return AgentResult(text="{}", model=spec.model)
        res = self._queue.pop(0)
        res.model = res.model or spec.model
        return res


class CallbackAgentRunner:
    """Fake runner that computes each response from the StageSpec (stage + prompt).

    Useful for the multi-agent DAG where per-ticker stages run concurrently and call order is
    nondeterministic. ``responder(spec)`` may return an AgentResult, a dict, or a JSON string.
    """

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[StageSpec] = []

    async def run(self, spec: StageSpec, run_id: str) -> AgentResult:
        self.calls.append(spec)
        out = self.responder(spec)
        if isinstance(out, AgentResult):
            out.model = out.model or spec.model
            return out
        if isinstance(out, dict):
            return AgentResult(text=json.dumps(out), model=spec.model)
        return AgentResult(text=str(out), model=spec.model)


# --- Semantic validation --------------------------------------------------------------------

_SOURCE_REF_RE = re.compile(r"^(?P<acc>[^#]+)#(?P<item>[^¶]+)¶(?P<idx>\d+)$")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def resolve_passage(source_ref: str) -> str | None:
    """Resolve a ``{accession}#{item}¶{idx}`` ref to its cached passage text, or None."""
    m = _SOURCE_REF_RE.match(source_ref.strip())
    if not m:
        return None
    text = sections.load_section_text(m["acc"], m["item"].upper())
    if text is None:
        return None
    passages = text.split("\n\n")
    idx = int(m["idx"])
    if 0 <= idx < len(passages):
        return passages[idx]
    return text  # fall back to whole-section text if the index drifted


def quote_matches(quote: str, passage: str, *, threshold: float = 0.9) -> bool:
    """True if the quote appears in the passage (fuzzy, whitespace-normalized)."""
    q, p = _normalize_ws(quote), _normalize_ws(passage)
    if not q:
        return False
    if q in p:
        return True
    match = SequenceMatcher(None, q, p).find_longest_match(0, len(q), 0, len(p))
    return (match.size / len(q)) >= threshold


def validate_citations(artifact: BaseModel) -> list[str]:
    """Every Citation.source_ref must resolve and its quote must match the passage."""
    errors: list[str] = []
    for cite in _iter_citations(artifact):
        passage = resolve_passage(cite.source_ref)
        if passage is None:
            errors.append(
                f"Citation source_ref does not resolve to a cached passage: {cite.source_ref!r}"
            )
            continue
        if not quote_matches(cite.quote, passage):
            errors.append(
                f"Quote not found in cited passage {cite.source_ref!r}: {cite.quote[:60]!r}"
            )
    return errors


def _iter_citations(artifact: BaseModel):
    from desk.contracts.v1 import Challenge, Citation, Claim

    for attr in ("claims", "bull_case"):
        for claim in getattr(artifact, attr, []) or []:
            if isinstance(claim, Claim):
                yield from claim.citations
    for attr in ("challenges", "bear_case"):
        for ch in getattr(artifact, attr, []) or []:
            if isinstance(ch, Challenge):
                yield from ch.citations
    # A bare list of citations (unlikely) — defensive.
    for c in getattr(artifact, "citations", []) or []:
        if isinstance(c, Citation):
            yield c


def validate_tickers_in_universe(artifact: BaseModel) -> list[str]:
    valid = set(universe_mod.universe_tickers())
    errors: list[str] = []
    tickers: list[str] = []
    if hasattr(artifact, "ticker") and artifact.ticker:
        tickers.append(artifact.ticker)
    for cand in getattr(artifact, "candidates", []) or []:
        tickers.append(getattr(cand, "ticker", ""))
    for t in tickers:
        if t and t.upper() not in valid:
            errors.append(f"Ticker not in universe: {t!r}")
    return errors


# --- JSON extraction ------------------------------------------------------------------------


def extract_json(text: str) -> dict | None:
    """Best-effort: parse the first complete JSON object in the agent's output text."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # Bracket-match the first {...}.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# --- AgentStage -----------------------------------------------------------------------------


class AgentStage:
    """Runs one agent stage and returns a validated, cost-stamped contract artifact."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        system_prompt: str,
        runner: AgentRunner,
        run_id: str,
        allowed_tools: list[str] | None = None,
        mcp_servers: dict[str, Any] | None = None,
        max_turns: int = 12,
        max_section_chars: int = 12_000,
        degradations: list[str] | None = None,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.runner = runner
        self.run_id = run_id
        self.allowed_tools = allowed_tools or []
        self.mcp_servers = mcp_servers or {}
        self.max_turns = max_turns
        self.max_section_chars = max_section_chars
        self.degradations = degradations or []

    async def run(
        self,
        prompt: str,
        *,
        contract_cls: type[Artifact],
        ticker: str | None = None,
        semantic_validators=None,
        output_schema: dict | None = None,
        inject: dict | None = None,
    ) -> Artifact:
        """Run the stage; return a validated artifact or raise HandoffFailure."""
        semantic_validators = semantic_validators or []
        inject = inject or {}
        token = set_context(
            ToolContext(
                run_id=self.run_id,
                stage=self.name,
                ticker=ticker,
                max_section_chars=self.max_section_chars,
            )
        )
        try:
            errors, raw = await self._attempt(
                prompt, contract_cls, ticker, semantic_validators, inject
            )
            if not errors:
                return raw  # type: ignore[return-value]
            # One repair retry with the validation errors appended.
            repair_prompt = (
                f"{prompt}\n\nYour previous response failed validation with these errors:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\n\nReturn corrected JSON only, matching the required schema exactly."
            )
            errors2, raw2 = await self._attempt(
                repair_prompt, contract_cls, ticker, semantic_validators, inject
            )
            if not errors2:
                return raw2  # type: ignore[return-value]
            raise HandoffFailure(self.name, errors2, str(raw2))
        finally:
            reset_context(token)

    async def _attempt(self, prompt, contract_cls, ticker, semantic_validators, inject=None):
        spec = StageSpec(
            stage=self.name,
            model=self.model,
            system_prompt=self.system_prompt,
            prompt=prompt,
            allowed_tools=self.allowed_tools,
            mcp_servers=self.mcp_servers,
            max_turns=self.max_turns,
        )
        result = await self.runner.run(spec, self.run_id)
        cost = self._log_and_cost(result, ticker)

        payload = result.structured_output or extract_json(result.text)
        if payload is None:
            return ([f"Output was not valid JSON: {result.text[:200]!r}"], result.text)

        # Inject provenance and any fields the agent can't supply (e.g. cost, disclaimer).
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("produced_by", f"{self.name} ({self.model})")
        for key, value in (inject or {}).items():
            payload.setdefault(key, value)

        try:
            artifact = contract_cls.model_validate(payload)
        except ValidationError as exc:
            return ([_fmt_pydantic_error(e) for e in exc.errors()], result.text)

        errors: list[str] = []
        for validator in semantic_validators:
            errors.extend(validator(artifact))
        if errors:
            return (errors, artifact)

        artifact.token_cost = cost
        return ([], artifact)

    def _log_and_cost(self, result: AgentResult, ticker: str | None) -> StageCost:
        u = result.usage
        computed = db.record_llm_call(
            run_id=self.run_id,
            stage=self.name,
            model=result.model or self.model,
            ticker=ticker,
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cache_read_tokens=u.get("cache_read_tokens", 0),
            cache_write_tokens=u.get("cache_write_tokens", 0),
            reported_cost_usd=result.reported_cost_usd,
            latency_ms=result.latency_ms,
            n_turns=result.n_turns,
            degradations=self.degradations,
        )
        return StageCost(
            stage=self.name,
            model=result.model or self.model,
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cache_read_tokens=u.get("cache_read_tokens", 0),
            cache_write_tokens=u.get("cache_write_tokens", 0),
            computed_cost_usd=computed,
            reported_cost_usd=result.reported_cost_usd,
            latency_ms=result.latency_ms,
            n_turns=result.n_turns,
            degradations=list(self.degradations),
        )


def _fmt_pydantic_error(err: dict) -> str:
    loc = ".".join(str(x) for x in err.get("loc", []))
    return f"{loc}: {err.get('msg', 'invalid')}"
