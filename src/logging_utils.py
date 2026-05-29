from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
SEED = 42


def ensure_dirs() -> None:
    RESULTS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)


def now_seconds() -> float:
    return time.perf_counter()


def elapsed(start: float) -> float:
    return time.perf_counter() - start


def json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    ensure_dirs()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dirs()
    rows = list(rows)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def env_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "seed": SEED,
    }


def ct_info(ct: Any) -> dict[str, Any]:
    return {
        "type": type(ct).__name__,
        "level": getattr(ct, "level", None),
        "nbytes": getattr(ct, "nbytes", None),
        "serialized_nbytes": getattr(ct, "serialized_nbytes", None),
    }


def exception_record(exc: BaseException, **context: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        **context,
    }
