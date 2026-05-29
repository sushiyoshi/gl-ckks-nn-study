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


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
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


def encrypted_row(condition: str, path: Path) -> dict[str, Any] | None:
    meta = read_json(path)
    if meta is None:
        return None
    timing = meta.get("timing_log") or {}
    return {
        "condition": condition,
        "same_task": condition == "gl_encrypted_weight_mlp_450",
        "comparison_role": "main" if condition == "gl_encrypted_weight_mlp_450" else "auxiliary",
        "n_samples": meta.get("n_samples"),
        "semantic_validation_passed": meta.get("semantic_validation_passed"),
        "task_type": meta.get("task_type"),
        "weight_privacy": meta.get("weight_privacy", "encrypted_weight"),
        "activation": meta.get("activation"),
        "runtime_total_s": meta.get("runtime_total_s"),
        "runtime_per_sample_s": meta.get("runtime_per_sample_s"),
        "server_only_s": first_float(
            timing.get("matrix_multiply_s"),
            (timing.get("linear1_matrix_multiply_s") or 0) + (timing.get("linear2_matrix_multiply_s") or 0) + (timing.get("activation_s") or 0)
            if timing
            else None,
        ),
        "relative_l2": first_float(meta.get("hidden_relative_l2"), meta.get("logits_relative_l2")),
        "accuracy": meta.get("accuracy"),
        "argmax_agreement": meta.get("argmax_agreement"),
        "failure_reason": meta.get("exception"),
        "failure_category": meta.get("failure_category"),
    }


def plaintext_row() -> dict[str, Any] | None:
    meta = read_json(RESULTS / "gl_padded_pca32_packed_results_n450.json")
    if meta:
        err = meta.get("error_metrics") or {}
        timing = meta.get("timing_log") or {}
        server_only = sum(float(timing.get(k) or 0.0) for k in ("linear1_add_s", "activation_s", "linear2_add_s"))
        return {
            "condition": "gl_plaintext_weight_pca32_packed",
            "same_task": True,
            "comparison_role": "main",
            "n_samples": meta.get("n_samples", 450),
            "semantic_validation_passed": meta.get("semantic_validation_passed"),
            "task_type": "mlp_poly_relu",
            "weight_privacy": "plaintext_weight",
            "activation": "degree3_poly",
            "runtime_total_s": meta.get("runtime_total_s"),
            "runtime_per_sample_s": meta.get("runtime_per_sample_amortized_s"),
            "server_only_s": server_only,
            "relative_l2": err.get("logits_relative_l2"),
            "accuracy": meta.get("accuracy_gl_decrypted"),
            "argmax_agreement": meta.get("argmax_agreement_poly_vs_gl"),
            "failure_reason": meta.get("exception"),
            "failure_category": None,
        }
    rows = read_csv_rows(RESULTS / "gl_pca32_packed_sweep.csv")
    rows450 = [r for r in rows if r.get("n_samples") == "450"]
    if not rows450:
        return None
    r = rows450[-1]
    return {
        "condition": "gl_plaintext_weight_pca32_packed",
        "same_task": True,
        "comparison_role": "main",
        "n_samples": r.get("n_samples"),
        "semantic_validation_passed": r.get("semantic_validation_passed"),
        "task_type": "mlp_poly_relu",
        "weight_privacy": "plaintext_weight",
        "activation": "degree3_poly",
        "runtime_total_s": r.get("runtime_total_s"),
        "runtime_per_sample_s": r.get("runtime_per_sample_s"),
        "server_only_s": None,
        "relative_l2": r.get("relative_l2"),
        "accuracy": r.get("accuracy"),
        "argmax_agreement": r.get("argmax_agreement_poly_vs_fhe"),
        "failure_reason": r.get("exception"),
        "failure_category": None,
    }


def md_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        "condition",
        "same_task",
        "comparison_role",
        "n_samples",
        "semantic_validation_passed",
        "task_type",
        "weight_privacy",
        "activation",
        "runtime_total_s",
        "runtime_per_sample_s",
        "relative_l2",
        "argmax_agreement",
        "failure_category",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows: list[dict[str, Any]] = []
    plain = plaintext_row()
    if plain:
        rows.append(plain)
    for condition, filename in [
        ("gl_encrypted_weight_mlp_450", "gl_encrypted_weight_pca32_mlp_n450.json"),
        ("gl_encrypted_weight_linear_450", "gl_encrypted_weight_pca32_linear_n450.json"),
        ("gl_encrypted_weight_two_linear_450", "gl_encrypted_weight_pca32_two_linear_n450.json"),
        ("gl_encrypted_weight_mlp_1", "gl_encrypted_weight_pca32_mlp_n1.json"),
        ("gl_encrypted_weight_mlp_32", "gl_encrypted_weight_pca32_mlp_n32.json"),
    ]:
        row = encrypted_row(condition, RESULTS / filename)
        if row:
            rows.append(row)
    write_csv(RESULTS / "gl_plain_vs_encrypted_weight_comparison.csv", rows)
    (RESULTS / "gl_plain_vs_encrypted_weight_comparison.md").write_text(md_table(rows))
    print(RESULTS / "gl_plain_vs_encrypted_weight_comparison.csv")


if __name__ == "__main__":
    main()
