"""Load + validate the screener evaluation suite and the dimension map."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_HERE = Path(__file__).resolve().parent
DEFAULT_SUITE = _HERE.parent / "data" / "screener_suite.yaml"
DIMENSION_MAP = _HERE / "dimension_map.yaml"


class Perturbation(BaseModel):
    change: dict[str, str]
    text: str
    expect_moved_fields: list[str]


class SuiteQuery(BaseModel):
    id: str
    text: str
    dimensions: dict[str, str] = Field(default_factory=dict)
    golden_filters: list[dict[str, Any]] = Field(default_factory=list)
    expected_tickers: list[str] = Field(default_factory=list)
    acceptable_extra: list[str] = Field(default_factory=list)
    paraphrases: list[str] = Field(default_factory=list)
    perturbations: list[Perturbation] = Field(default_factory=list)
    starve: bool = False
    expected_relaxation_field: str | None = None


def load_suite(path: Path | str = DEFAULT_SUITE) -> list[SuiteQuery]:
    data = yaml.safe_load(Path(path).read_text("utf-8"))
    return [SuiteQuery.model_validate(q) for q in data.get("queries", [])]


def load_dimension_map(path: Path | str = DIMENSION_MAP) -> dict[str, list[str]]:
    data = yaml.safe_load(Path(path).read_text("utf-8"))
    return data.get("dimensions", {})
