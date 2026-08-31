"""Strict JSON conversion and atomic-write helpers."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert scientific Python objects to RFC-compliant JSON values."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def dump_json(payload: Any, handle, *, indent: int | None = 2) -> None:
    json.dump(json_safe(payload), handle, indent=indent, allow_nan=False)


def dump_json_atomic(payload: Any, path: str | Path, *, indent: int | None = 2) -> None:
    """Write JSON beside its destination, then atomically replace it."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            dump_json(payload, handle, indent=indent)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["dump_json", "dump_json_atomic", "json_safe"]
