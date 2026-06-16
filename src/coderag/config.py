"""Configuration loading.

A single ``config.yaml`` at the repository root holds every tunable parameter
(model names, retrieval depth, chunk sizes, codebase locations). Relative paths
in the config are resolved against the repository root so the project works the
same regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# src/coderag/config.py -> repo root is two levels up from this file's parent.
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CONFIG: dict[str, Any] = load_config()


def resolve_path(p: str | Path) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def codebase_config(name: str) -> dict[str, Any]:
    try:
        return CONFIG["codebases"][name]
    except KeyError as exc:  # pragma: no cover - guard for typos in callers
        known = ", ".join(CONFIG.get("codebases", {}))
        raise KeyError(f"Unknown codebase '{name}'. Known: {known}") from exc
