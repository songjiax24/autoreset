"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from seed_preview_cv.common.paths import PROJECT_ROOT


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of {path}")
    return data


def resolve_path(path: Path | str, base: Path | None = None) -> Path:
    """Resolve a path relative to project root (or an explicit base)."""
    path = Path(path)
    if path.is_absolute():
        return path
    root = base if base is not None else PROJECT_ROOT
    return (root / path).resolve()
