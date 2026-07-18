"""Runtime configuration: environment variables + YAML config files.

Environment comes from ``pydantic-settings`` (``.env`` supported). YAML config files
(``config/*.yaml``) are loaded lazily and cached. Keeping both behind one module means the
rest of the codebase never reads ``os.environ`` or opens YAML directly.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _repo_root() -> Path:
    # src/desk/settings.py -> repo root is three parents up.
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-driven settings. Values load from the process env or a local ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    sec_edgar_user_agent: str = Field(
        default="research-desk (set SEC_EDGAR_USER_AGENT)",
        alias="SEC_EDGAR_USER_AGENT",
    )
    desk_cache_dir: str = Field(default=".cache", alias="DESK_CACHE_DIR")
    desk_runs_dir: str = Field(default="runs", alias="DESK_RUNS_DIR")

    def _resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else _repo_root() / p

    @property
    def cache_dir(self) -> Path:
        return self._resolve(self.desk_cache_dir)

    @property
    def config_dir(self) -> Path:
        return _repo_root() / "config"

    @property
    def runs_dir(self) -> Path:
        return self._resolve(self.desk_runs_dir)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@functools.cache
def load_yaml_config(name: str) -> dict[str, Any]:
    """Load and cache a YAML file from ``config/`` by base name (e.g. ``"models"``)."""
    path = get_settings().config_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def reset_caches() -> None:
    """Clear cached settings/configs — used by tests that patch env or config files."""
    get_settings.cache_clear()
    load_yaml_config.cache_clear()
