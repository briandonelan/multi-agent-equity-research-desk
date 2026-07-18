"""Shared helpers for comparing the filters a screener applied."""

from __future__ import annotations

from desk.contracts.v1 import ScreenFilter


def _norm_value(value) -> object:
    """Normalize a filter value so equal filters compare equal regardless of list/tuple."""
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value)
    if isinstance(value, bool):  # guard: bool is an int subclass
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def filters_by_field(filters: list[ScreenFilter]) -> dict[str, tuple[str, object]]:
    """Index filters by field as ``field -> (op, normalized_value)``.

    When a field is filtered more than once (e.g. a ``>=``/``<`` pair bounding market cap), the
    entries are folded into one comparable signature so a change on either bound is detected.
    """
    out: dict[str, tuple[str, object]] = {}
    for f in filters:
        sig = (f.op, _norm_value(f.value))
        if f.field in out:
            # Combine repeated filters on the same field into an order-independent signature.
            prev = out[f.field]
            merged = tuple(sorted([prev, sig], key=repr))
            out[f.field] = ("multi", merged)
        else:
            out[f.field] = sig
    return out


def changed_fields(
    base: list[ScreenFilter], other: list[ScreenFilter]
) -> set[str]:
    """Fields whose filter signature differs between two runs (added, dropped, or altered)."""
    a = filters_by_field(base)
    b = filters_by_field(other)
    changed: set[str] = set()
    for field in set(a) | set(b):
        if a.get(field) != b.get(field):
            changed.add(field)
    return changed
