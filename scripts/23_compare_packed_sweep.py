from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, write_csv


def read_rows(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def f(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def main() -> None:
    rows = []
    for source in [
        "ckks_pca32_packed_sweep.csv",
        "ckks_pca32_packed_sweep_plainmatrix.csv",
        "ckks_pca32_packed_sweep_lightplainmatrix.csv",
        "gl_pca32_packed_sweep.csv",
    ]:
        rows.extend(read_rows(source))
    by_n = {}
    for row in rows:
        by_n.setdefault(row.get("n_samples"), []).append(row)
    out = []
    for n, group in by_n.items():
        dense = next((r for r in group if r.get("condition") == "ckks_pca32_packed_dense"), None)
        gl = next((r for r in group if r.get("condition") == "gl_pca32_packed"), None)
        dense_time = f(dense or {}, "runtime_per_sample_s")
        gl_time = f(gl or {}, "runtime_per_sample_s")
        for row in group:
            item = {
                "condition": row.get("condition"),
                "n_samples": row.get("n_samples"),
                "semantic_validation_passed": row.get("semantic_validation_passed"),
                "accuracy": row.get("accuracy"),
                "relative_l2": row.get("relative_l2"),
                "ciphertext_count": row.get("ciphertext_count"),
                "slot_or_matrix_utilization": row.get("slot_utilization") or row.get("slot_or_matrix_utilization"),
                "matrix_build_s": row.get("matrix_build_s"),
                "key_generation_s": row.get("key_generation_s"),
                "encryption_s_total": row.get("encryption_s_total"),
                "linear1_s_total": row.get("linear1_s_total"),
                "activation_s_total": row.get("activation_s_total"),
                "linear2_s_total": row.get("linear2_s_total"),
                "decryption_s_total": row.get("decryption_s_total"),
                "runtime_total_s": row.get("runtime_total_s"),
                "runtime_per_sample_s": row.get("runtime_per_sample_s"),
            }
            row_time = f(row, "runtime_per_sample_s")
            item["speedup_vs_ckks_dense"] = dense_time / row_time if dense_time and row_time else None
            item["speedup_vs_gl"] = gl_time / row_time if gl_time and row_time else None
            out.append(item)
    out.sort(key=lambda r: (int(r["n_samples"]), r["condition"] or ""))
    write_csv(RESULTS / "packed_sweep_comparison.csv", out)
    print(RESULTS / "packed_sweep_comparison.csv")


if __name__ == "__main__":
    main()
