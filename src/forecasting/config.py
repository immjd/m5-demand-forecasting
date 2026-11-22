"""Load the project configuration from ``configs/config.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config and return it as a plain dict.

    Args:
        path: Optional path to a YAML file. Defaults to ``configs/config.yaml``
            at the project root.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Return an absolute path for an entry in ``config['paths']``."""
    return PROJECT_ROOT / config["paths"][key]
