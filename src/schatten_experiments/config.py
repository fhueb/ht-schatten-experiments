from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return loaded


def parse_scalar(value: str) -> Any:
    """Parse CLI key=value strings into simple YAML-like scalar values."""
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "inf":
        return "inf"
    try:
        if any(ch in value for ch in (".", "e", "E")):
            parsed = float(value)
            return math.inf if math.isinf(parsed) else parsed
        return int(value)
    except ValueError:
        return value


def parse_key_value_items(items: list[str] | None) -> dict[str, Any]:
    """Parse repeated KEY=VALUE items from CLI arguments."""
    parsed: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE item, got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty override key in: {item!r}")
        parsed[key] = parse_scalar(value.strip())
    return parsed


def apply_overwrites(
    config: dict[str, Any],
    overwrites: dict[str, Any],
    *,
    default_section: str | None = None,
) -> dict[str, Any]:
    """Apply CLI overwrites, supporting dotted keys for nested config values."""
    out = copy.deepcopy(config)
    for key, value in overwrites.items():
        path = key.split(".")
        if len(path) == 1:
            if default_section is None:
                raise ValueError(f"Overwrite {key!r} needs a section prefix.")
            path = [default_section, key]

        target = out
        for part in path[:-1]:
            existing = target.get(part)
            if existing is None:
                existing = {}
            if not isinstance(existing, dict):
                raise ValueError(f"Cannot override through non-mapping key: {part!r}")
            target[part] = dict(existing)
            target = target[part]
        target[path[-1]] = value
    return out


def apply_training_overwrites(config: dict[str, Any], overwrites: dict[str, Any]) -> dict[str, Any]:
    """Apply flat CLI overwrites to the training section of a config."""
    return apply_overwrites(config, overwrites, default_section="training")
