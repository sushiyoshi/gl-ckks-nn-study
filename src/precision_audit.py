from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .logging_utils import ensure_dirs, json_default


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _level(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        levels = [getattr(item, "level", None) for item in value]
        present = [level for level in levels if level is not None]
        if not present:
            return None
        if len(set(present)) == 1:
            return present[0]
        return present
    return getattr(value, "level", None)


def tensor_error_metrics(name: str, plain: Any, encrypted_or_gl: Any) -> dict[str, Any]:
    expected = _array(plain)
    actual = _array(encrypted_or_gl)
    if expected.shape != actual.shape:
        raise ValueError(f"{name} shape mismatch: plain={expected.shape}, actual={actual.shape}")

    err = actual - expected
    plain_norm = float(np.linalg.norm(expected.ravel()))
    err_norm = float(np.linalg.norm(err.ravel()))
    denom = plain_norm if plain_norm > 0.0 else 1.0
    finite_actual = actual[np.isfinite(actual)]
    finite_err = err[np.isfinite(err)]
    finite_plain = expected[np.isfinite(expected)]

    return {
        "stage": name,
        "shape": list(expected.shape),
        "level": _level(encrypted_or_gl),
        "relative_l2": err_norm / denom,
        "linf": float(np.max(np.abs(finite_err))) if finite_err.size else None,
        "mae": float(np.mean(np.abs(finite_err))) if finite_err.size else None,
        "rms_plain": float(np.sqrt(np.mean(finite_plain * finite_plain))) if finite_plain.size else None,
        "rms_error": float(np.sqrt(np.mean(finite_err * finite_err))) if finite_err.size else None,
        "max_abs_plain": float(np.max(np.abs(finite_plain))) if finite_plain.size else None,
        "max_abs_error": float(np.max(np.abs(finite_err))) if finite_err.size else None,
        "allclose_rtol_1e-5_atol_1e-6": bool(np.allclose(actual, expected, rtol=1e-5, atol=1e-6)),
        "allclose_rtol_1e-4_atol_1e-5": bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-5)),
        "nan_count": int(np.isnan(actual).sum()),
        "inf_count": int(np.isinf(actual).sum()),
    }


def summarize_array(name: str, arr: Any) -> dict[str, Any]:
    value = _array(arr)
    finite = value[np.isfinite(value)]
    return {
        "stage": name,
        "shape": list(value.shape),
        "level": _level(arr),
        "rms_plain": float(np.sqrt(np.mean(finite * finite))) if finite.size else None,
        "max_abs_plain": float(np.max(np.abs(finite))) if finite.size else None,
        "nan_count": int(np.isnan(value).sum()),
        "inf_count": int(np.isinf(value).sum()),
    }


def _csv_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("stages", []):
        flat = dict(row)
        for key, value in list(flat.items()):
            if isinstance(value, (list, dict)):
                flat[key] = json.dumps(value, ensure_ascii=False, default=json_default)
        rows.append(flat)
    return rows


def write_precision_audit(path: str | Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    ensure_dirs()
    json_path = Path(path)
    if json_path.suffix != ".json":
        json_path = json_path.with_suffix(".json")
    csv_path = json_path.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n")

    rows = _csv_rows(payload)
    if not rows:
        csv_path.write_text("")
        return json_path, csv_path

    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path
