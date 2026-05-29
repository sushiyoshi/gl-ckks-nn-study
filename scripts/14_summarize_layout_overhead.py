from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, write_csv


def read_json(path: str) -> dict:
    p = RESULTS / path
    return json.loads(p.read_text()) if p.exists() else {}


def read_csv_first(path: str) -> dict:
    p = RESULTS / path
    if not p.exists() or p.stat().st_size == 0:
        return {}
    with p.open() as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def flatten_error(meta: dict) -> dict:
    err = meta.get("error_metrics") or {}
    return {
        "logits_linf": err.get("logits_linf") or err.get("fhe_numeric_logits_linf"),
        "logits_mae": err.get("logits_mae") or err.get("fhe_numeric_logits_mae"),
        "logits_relative_l2": err.get("logits_relative_l2") or err.get("fhe_numeric_logits_relative_l2"),
    }


def row(name: str, meta: dict, notes: str = "") -> dict:
    ops = meta.get("operation_counts") or {}
    return {
        "condition": name,
        "ok": meta.get("ok", bool(meta)),
        "semantic_validation_passed": meta.get("semantic_validation_passed"),
        "logical_dims": meta.get("logical_dims"),
        "physical_dim": meta.get("physical_dim"),
        "slot_count": meta.get("slot_count"),
        "used_slots": meta.get("used_slots"),
        "packing_utilization": meta.get("packing_utilization"),
        "matrix_multiply_count": ops.get("matrix_multiply") or meta.get("matrix_multiply_count"),
        "add_count": ops.get("add") or meta.get("add_count"),
        "polynomial_degree": meta.get("polynomial_degree"),
        "argmax_agreement": meta.get("argmax_agreement") or meta.get("argmax_agreement_poly_vs_gl") or meta.get("argmax_agreement_poly_vs_ckks"),
        "accuracy": meta.get("accuracy_gl_decrypted") or meta.get("accuracy_ckks_decrypted") or meta.get("accuracy_fhe_decrypted"),
        "output_padding_ratio": meta.get("output_padding_ratio"),
        "hidden_padding_ratio": meta.get("hidden_padding_ratio"),
        "input_block_count": meta.get("input_block_count"),
        **flatten_error(meta),
        "notes": notes,
    }


def main() -> None:
    plain = read_csv_first("plain_baseline.csv")
    rows = [
        {
            "condition": "plain_digits64_baselines",
            "ok": bool(plain),
            "accuracy_relu_plain": plain.get("accuracy_original_relu"),
            "accuracy_poly_plain": plain.get("accuracy_poly_plain"),
            "argmax_agreement": plain.get("argmax_agreement_relu_poly"),
            "notes": "64->16->10 plaintext baselines",
        },
        row("ckks_digits64_poly", read_json("ckks_results.json"), "existing 64->16->10 CKKS baseline"),
        row("ckks_pca32_poly", read_json("ckks_pca32_results.json"), "PCA32 32->32->10 CKKS baseline"),
        row("gl_native_toy_32", read_json("gl_native_toy.json"), "synthetic 32->32->32 native GL layout"),
        row("gl_padded_pca32", read_json("gl_padded_pca32_results.json"), "PCA32 32->32->10 with 22 padded output rows"),
        row("gl_blocked_digits64", read_json("gl_blocked_digits64_results.json"), "64->16->10 decomposed into 32x32 GL blocks"),
    ]
    write_csv(RESULTS / "layout_overhead_summary.csv", rows)
    print(RESULTS / "layout_overhead_summary.csv")


if __name__ == "__main__":
    main()
