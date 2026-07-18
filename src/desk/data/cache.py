"""On-disk cache for network responses and extracted artifacts.

Two access patterns:

- **Keyed cache with TTL** (``get_json`` / ``set_json``, ``get_text`` / ``set_text``): logical
  keys namespaced by source, with a per-entry TTL. Used for EDGAR submissions / companyfacts and
  yfinance metrics that should refresh daily but stay offline within a run.
- **Content-addressed store** (``store_blob`` / ``load_blob``): immutable artifacts (filing HTML,
  extracted section text) keyed by a caller-supplied stable key, cached forever.

Everything lives under ``<cache_dir>/<namespace>/``. The cache is intentionally simple and
filesystem-based; no locking is needed because writes are whole-file and idempotent.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from desk.settings import get_settings

# TTLs in seconds, by logical source. `None` means cache forever.
TTL_SECONDS: dict[str, int | None] = {
    "company_tickers": 7 * 24 * 3600,  # SEC ticker map: 7 days
    "submissions": 24 * 3600,  # filings index: 1 day
    "companyfacts": 24 * 3600,  # XBRL facts: 1 day
    "yfinance": 24 * 3600,  # prices/ratios: 1 day
    "metrics": 24 * 3600,  # derived metrics table: 1 day
    "filing": None,  # raw filing docs: forever (content-addressed)
    "section": None,  # extracted sections: forever
}


def _cache_root() -> Path:
    return get_settings().cache_dir


def _safe_name(key: str) -> str:
    """Turn an arbitrary logical key into a filesystem-safe, collision-resistant name."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    # Keep a readable prefix for debuggability.
    prefix = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:48]
    return f"{prefix}-{digest}"


def _entry_path(namespace: str, key: str, suffix: str) -> Path:
    d = _cache_root() / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{_safe_name(key)}{suffix}"


def _is_fresh(namespace: str, meta_path: Path) -> bool:
    ttl = TTL_SECONDS.get(namespace)
    if ttl is None:
        return meta_path.exists()  # forever, as long as it exists
    try:
        meta = json.loads(meta_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    fetched_at = meta.get("fetched_at", 0)
    return (time.time() - fetched_at) < ttl


# --- Keyed cache with TTL -------------------------------------------------------------------


def get_text(namespace: str, key: str) -> str | None:
    """Return cached text if present and fresh, else None."""
    data_path = _entry_path(namespace, key, ".data")
    meta_path = _entry_path(namespace, key, ".meta.json")
    if not data_path.exists() or not _is_fresh(namespace, meta_path):
        return None
    try:
        return data_path.read_text("utf-8")
    except OSError:
        return None


def set_text(namespace: str, key: str, value: str, *, origin: str | None = None) -> None:
    data_path = _entry_path(namespace, key, ".data")
    meta_path = _entry_path(namespace, key, ".meta.json")
    data_path.write_text(value, "utf-8")
    meta_path.write_text(
        json.dumps({"fetched_at": time.time(), "key": key, "origin": origin}),
        "utf-8",
    )


def get_json(namespace: str, key: str) -> Any | None:
    text = get_text(namespace, key)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def set_json(namespace: str, key: str, value: Any, *, origin: str | None = None) -> None:
    set_text(namespace, key, json.dumps(value), origin=origin)


# --- Content-addressed immutable store ------------------------------------------------------


def store_blob(namespace: str, key: str, value: str) -> str:
    """Store an immutable text blob under a stable key. Returns the on-disk path."""
    path = _entry_path(namespace, key, ".data")
    meta_path = _entry_path(namespace, key, ".meta.json")
    path.write_text(value, "utf-8")
    if not meta_path.exists():
        meta_path.write_text(json.dumps({"fetched_at": time.time(), "key": key}), "utf-8")
    return str(path)


def load_blob(namespace: str, key: str) -> str | None:
    path = _entry_path(namespace, key, ".data")
    if not path.exists():
        return None
    try:
        return path.read_text("utf-8")
    except OSError:
        return None
