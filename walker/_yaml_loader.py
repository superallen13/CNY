from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_RESERVED_KEYS = {"_doc", "extends"}

def deep_merge(base: dict, leaf: dict) -> dict:
    
    out = dict(base)
    for k, v in leaf.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def resolve_extends(path: Path, seen: list[Path] | None = None) -> dict:
    
    seen = seen or []
    abs_path = path.resolve()
    if abs_path in seen:
        chain = " → ".join(str(p) for p in seen + [abs_path])
        raise ValueError(f"circular extends: {chain}")
    seen = seen + [abs_path]

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a mapping")

    parent_name = raw.pop("extends", None)
    if parent_name is not None:
        if not isinstance(parent_name, str):
            raise ValueError(
                f"{path}: `extends` must be a string, got {type(parent_name).__name__}"
            )
        if not (parent_name.endswith(".yaml") or parent_name.endswith(".yml")):
            parent_name = parent_name + ".yaml"
        parent_path = path.parent / parent_name
        if not parent_path.exists():
            raise ValueError(f"{path}: parent yaml not found: {parent_path}")
        parent = resolve_extends(parent_path, seen)
    else:
        parent = {}

    raw.pop("_doc", None)
    return deep_merge(parent, raw)

def coerce(s: str) -> Any:
    
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    if s.startswith("[") or s.startswith("{"):
        parsed = yaml.safe_load(s)
        if isinstance(parsed, (list, dict)):
            return parsed
        raise ValueError(f"override value is not valid yaml: {s!r}")
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

def set_nested(d: dict, path: list[str], val: Any) -> None:
    
    cur = d
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
        if not isinstance(cur, dict):
            raise ValueError(f"override path collides with non-dict: {'.'.join(path)}")
    cur[path[-1]] = val

def apply_overrides(raw: dict, overrides: list[str] | None) -> dict:
    
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be KEY=VAL, got {ov!r}")
        key, val = ov.split("=", 1)
        set_nested(raw, key.split("."), coerce(val))
    return raw
