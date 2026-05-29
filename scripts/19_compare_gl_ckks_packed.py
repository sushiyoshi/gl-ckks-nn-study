from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, write_csv


def load(name: str) -> dict:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def err(meta: dict, key: str) -> float | None:
    return (meta.get("error_metrics") or {}).get(key)


def row(condition: str, meta: dict) -> dict:
    ops = meta.get("operation_counts") or {}
    return {
        "condition": condition,
        "ok": meta.get("ok"),
        "semantic_validation_passed": meta.get("semantic_validation_passed"),
        "n_samples": meta.get("n_samples"),
        "accuracy": meta.get("accuracy") or meta.get("accuracy_gl_decrypted"),
        "argmax_agreement": meta.get("argmax_agreement_poly_vs_fhe") or meta.get("argmax_agreement_poly_vs_gl"),
        "logits_linf": err(meta, "logits_linf"),
        "logits_mae": err(meta, "logits_mae"),
        "logits_relative_l2": err(meta, "logits_relative_l2"),
        "slot_count": meta.get("slot_count"),
        "samples_per_ciphertext": meta.get("samples_per_ciphertext") or meta.get("samples_per_ciphertext_capacity"),
        "ciphertext_count": meta.get("ciphertext_count"),
        "sample_packing_utilization": meta.get("sample_packing_utilization"),
        "slot_or_entry_utilization": meta.get("slot_utilization") or meta.get("matrix_entry_utilization_input"),
        "matrix_multiply_count": meta.get("matrix_multiply_count") or ops.get("matrix_multiply"),
        "add_count": meta.get("add_count") or ops.get("add"),
        "runtime_total_s": meta.get("runtime_total_s"),
        "runtime_per_sample_amortized_s": meta.get("runtime_per_sample_amortized_s"),
        "output_padding_ratio": meta.get("output_padding_ratio"),
        "hidden_padding_ratio": meta.get("hidden_padding_ratio"),
    }


def main() -> None:
    rows = [
        row("gl_padded_pca32_packed_n450", load("gl_padded_pca32_packed_results_n450.json")),
        row("ckks_pca32_packed_n32", load("ckks_pca32_packed_results_n32.json")),
        row("ckks_pca32_packed_n450", load("ckks_pca32_packed_results_n450.json")),
        row("gl_blocked_digits64_packed_n450", load("gl_blocked_digits64_packed_results_n450.json")),
    ]
    gl = rows[0].get("runtime_per_sample_amortized_s")
    for item in rows:
        per = item.get("runtime_per_sample_amortized_s")
        if gl and per:
            item["runtime_per_sample_vs_gl_pca32_packed_450"] = per / gl
    write_csv(RESULTS / "gl_ckks_packed_comparison.csv", rows)
    print(RESULTS / "gl_ckks_packed_comparison.csv")


if __name__ == "__main__":
    main()
