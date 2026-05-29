from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, write_csv


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def first_float(*values: Any) -> float | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def gl_row() -> dict[str, Any] | None:
    probe = read_json(RESULTS / "gl_ctct_matrix_multiply_probe.json")
    if not probe:
        return None
    cases = probe.get("cases") or []
    ct_case = next((case for case in cases if case.get("case") == "ct_ct"), cases[0] if cases else None)
    if not ct_case:
        return None
    shape = probe.get("shape") or []
    d = int(shape[1]) if len(shape) > 1 else None
    return {
        "library": "GL",
        "scheme_or_backend": "GL ct-ct matrix_multiply",
        "d": d,
        "encrypted_operands": "ct_ct",
        "packing_layout": "batch x 32 x 32",
        "semantic_validation_passed": probe.get("semantic_validation_passed"),
        "relative_l2": ct_case.get("matrix_relative_l2"),
        "keygen_s": probe.get("key_generation_s"),
        "rotation_or_mm_keygen_s": None,
        "encryption_s": probe.get("encryption_s"),
        "matmul_s": ct_case.get("matrix_multiply_s"),
        "decryption_s": ct_case.get("decryption_s"),
        "runtime_total_s": probe.get("runtime_total_s"),
        "notes": "GL ct-ct probe is batch-oriented and not packing-matched with OpenFHE-NumPy",
    }


def openfhe_rows() -> list[dict[str, Any]]:
    rows = read_csv(RESULTS / "openfhe_numpy_ckks_matmul_sweep.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "library": "OpenFHE-NumPy",
                "scheme_or_backend": row.get("scheme_or_backend") or "CKKS",
                "d": int(row["d"]) if row.get("d") not in (None, "") else None,
                "encrypted_operands": row.get("encrypted_operands") or "ct_ct",
                "packing_layout": row.get("packing_layout"),
                "semantic_validation_passed": row.get("semantic_validation_passed"),
                "relative_l2": first_float(row.get("relative_l2")),
                "keygen_s": first_float(row.get("keygen_s")),
                "rotation_or_mm_keygen_s": first_float(row.get("rotation_or_mm_keygen_s")),
                "encryption_s": first_float(row.get("encryption_s")),
                "matmul_s": first_float(row.get("matmul_s")),
                "decryption_s": first_float(row.get("decryption_s")),
                "runtime_total_s": first_float(row.get("runtime_total_s")),
                "notes": row.get("notes") or row.get("exception") or row.get("unsupported_reason"),
            }
        )
    return out


def main() -> None:
    rows: list[dict[str, Any]] = []
    gl = gl_row()
    if gl:
        rows.append(gl)
    rows.extend(openfhe_rows())
    write_csv(RESULTS / "gl_vs_openfhe_ctct_matmul_comparison.csv", rows)
    print(RESULTS / "gl_vs_openfhe_ctct_matmul_comparison.csv")


if __name__ == "__main__":
    main()
