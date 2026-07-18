"""Disk cache: TTL freshness and the content-addressed store."""

from __future__ import annotations

import time

from desk.data import cache


def test_set_get_json_roundtrip():
    cache.set_json("metrics", "AAPL", {"a": 1})
    assert cache.get_json("metrics", "AAPL") == {"a": 1}


def test_ttl_expiry(monkeypatch):
    cache.set_json("metrics", "TSLA", {"x": 2})
    assert cache.get_json("metrics", "TSLA") == {"x": 2}
    # Force the entry to look old by advancing time past the 1-day TTL.
    real = time.time()
    monkeypatch.setattr(time, "time", lambda: real + 2 * 24 * 3600)
    assert cache.get_json("metrics", "TSLA") is None


def test_content_addressed_blob_is_permanent(monkeypatch):
    cache.store_blob("filing", "320193/acc/doc.htm", "<html>hi</html>")
    real = time.time()
    monkeypatch.setattr(time, "time", lambda: real + 999 * 24 * 3600)
    # "filing"/"section" namespaces have no TTL -> always fresh.
    assert cache.load_blob("filing", "320193/acc/doc.htm") == "<html>hi</html>"


def test_missing_returns_none():
    assert cache.get_json("submissions", "NOPE") is None
    assert cache.load_blob("filing", "nope") is None
