"""Per-run bookkeeping: run_id, artifact persistence, and a reproducibility manifest.

Every prompt, contract, degradation, and failure is written under ``runs/{run_id}/`` so any memo
can be reproduced. ``manifest.json`` snapshots the config (models, budgets, universe hash, git
SHA) that produced the run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel
from ulid import ULID

from desk.settings import get_settings, load_yaml_config


def new_run_id() -> str:
    return str(ULID())


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(get_settings().config_dir.parent),
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _universe_hash() -> str:
    cfg = load_yaml_config("universe")
    payload = json.dumps(cfg.get("tickers", []), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class RunContext:
    """Owns the on-disk layout for one run and the writers that populate it."""

    def __init__(self, run_id: str, *, engine: str, query: str):
        self.run_id = run_id
        self.engine = engine
        self.query = query
        self.root = get_settings().runs_dir / run_id
        for sub in ("handoffs", "prompts", "memos", "failures"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "run_id": self.run_id,
            "engine": self.engine,
            "query": self.query,
            "created_at": datetime.now(UTC).isoformat(),
            "git_sha": _git_sha(),
            "universe_hash": _universe_hash(),
            "models": load_yaml_config("models").get("stages", {}),
            "budgets": load_yaml_config("budgets"),
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")

    def write_prompt(self, stage: str, prompt: str) -> None:
        (self.root / "prompts" / f"{stage}.txt").write_text(prompt, "utf-8")

    def write_artifact(self, name: str, artifact: BaseModel) -> None:
        (self.root / "handoffs" / f"{name}.json").write_text(
            artifact.model_dump_json(indent=2), "utf-8"
        )

    def write_failure(self, name: str, record: dict) -> None:
        (self.root / "failures" / f"{name}.json").write_text(
            json.dumps(record, indent=2, default=str), "utf-8"
        )

    def write_memo(self, ticker: str, markdown: str, memo_json: BaseModel) -> Path:
        md_path = self.root / "memos" / f"{ticker}.md"
        md_path.write_text(markdown, "utf-8")
        (self.root / "memos" / f"{ticker}.json").write_text(
            memo_json.model_dump_json(indent=2), "utf-8"
        )
        return md_path

    def memo_paths(self) -> list[Path]:
        return sorted((self.root / "memos").glob("*.md"))

    def write_result(
        self, *, n_memos: int, n_failures: int, memo_tickers: list[str], failures: list[dict]
    ) -> None:
        """Write the run's terminal outcome. This file is the LAST thing a run writes, so its
        presence means the run finished: a run with ``result.json`` and ``n_memos == 0`` is a
        genuine empty screen, whereas a run missing ``result.json`` was killed mid-flight. It also
        records the authoritative failure count (the per-ticker ``failures/`` files can undercount
        pipeline-level errors)."""
        n_timeouts = sum(1 for f in failures if f.get("timeout"))
        payload = {
            "run_id": self.run_id,
            "engine": self.engine,
            "query": self.query,
            "status": "complete",
            "finished_at": datetime.now(UTC).isoformat(),
            "n_memos": n_memos,
            "n_failures": n_failures,
            "n_timeouts": n_timeouts,  # subset of failures that were wall-clock timeouts
            "memo_tickers": memo_tickers,
            "failures": [
                {
                    "stage": f.get("stage"),
                    "errors": f.get("errors", []),
                    "timeout": bool(f.get("timeout")),
                }
                for f in failures
            ],
        }
        (self.root / "result.json").write_text(json.dumps(payload, indent=2), "utf-8")
