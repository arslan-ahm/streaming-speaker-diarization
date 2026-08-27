"""Run artefacts: directories, JSONL history, JSON summaries, CSV tables.

Two rules encoded here because they were the source of real bugs elsewhere:

1. **LF line endings.** ``pathlib.Path.write_text`` uses the platform default,
   which on Windows is CRLF, which makes every committed CSV and JSON show as
   modified on a Linux checkout. Everything goes through :func:`write_text`.
2. **NumPy scalars are not JSON.** ``json.dumps(np.float32(1.0))`` raises. The
   encoder below coerces rather than letting a summary silently fail to write
   at the end of a 15-minute run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_text(path: str | Path, text: str) -> Path:
    """Write text with LF endings, creating parents."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return p


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    return write_text(path, json.dumps(_jsonable(payload), indent=indent) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one record. Used for training history so a killed run keeps its log."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(_jsonable(dict(record))) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]] | pd.DataFrame) -> Path:
    """Write a table with LF endings and a stable float format."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, lineterminator="\n")
    return p


def run_dir(root: str | Path, name: str) -> Path:
    d = Path(root) / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    return d
