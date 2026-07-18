"""Schema-version handling for handoff artifacts.

Only ``"1"`` exists today. The registry gives one place to add a v2 later and route incoming
artifacts to the right model set, so agents never hardcode a version.
"""

from __future__ import annotations

from desk.contracts import v1

CURRENT_VERSION = v1.SCHEMA_VERSION

# All contract models live in the v1 module. Schema evolutions that are additive and
# backward-compatible (e.g. v1.1 adding ScreenResult.relaxations) map to the same module — the
# models tolerate missing new fields, so an older artifact loads with the new fields defaulted.
_MODELS_BY_VERSION = {
    "1": v1,
    "1.1": v1,
}


def models_for(version: str = CURRENT_VERSION):
    """Return the module holding the contract models for a schema version."""
    if version not in _MODELS_BY_VERSION:
        raise ValueError(f"Unknown schema_version {version!r}; known: {list(_MODELS_BY_VERSION)}")
    return _MODELS_BY_VERSION[version]
