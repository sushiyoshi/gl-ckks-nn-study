from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, write_csv


def load(name: str) -> dict:
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def total_time(meta: dict) -> float | None:
    if "runtime_total_s" in meta:
        return meta["runtime_total_s"]
    timing = meta.get("timing_log")
    if isinstance(timing, dict):
        return sum(float(v) for v in timing.values() if isinstance(v, (int, float)))
    return None


def n_samples(meta: dict) -> int:
    return int(meta.get("n_samples") or meta.get("used_input_entries", 0) / max(1, meta.get("feature_dim", 1)) or 1)


def row(condition: str, meta: dict) -> dict:
    ops = meta.get("operation_counts") or {}
    t = total_time(meta)
    n = int(meta.get("n_samples") or meta.get("n_samples_requested") or meta.get("used_input_entries", 0) / max(1, meta.get("feature_dim", 1)) or 1)
    if meta.get("runtime_per_sample_amortized_s") is not None:
        per = meta["runtime_per_sample_amortized_s"]
    elif t is not None:
        per = t / n
    else:
        per = None
    err = meta.get("error_metrics") or {}
    return {
        "condition": condition,
        "ok": meta.get("ok"),
        "semantic_validation_passed": meta.get("semantic_validation_passed"),
        "n_samples": n,
        "runtime_total_s": t,
        "runtime_per_sample_amortized_s": per,
        "matrix_multiply_count": ops.get("matrix_multiply"),
        "add_count": ops.get("add"),
        "sample_packing_utilization": meta.get("sample_packing_utilization"),
        "matrix_entry_utilization_input": meta.get("matrix_entry_utilization_input"),
        "used_batches": meta.get("used_batches"),
        "used_columns_last_batch": meta.get("used_columns_last_batch"),
        "output_padding_ratio": meta.get("output_padding_ratio"),
        "hidden_padding_ratio": meta.get("hidden_padding_ratio"),
        "argmax_agreement": meta.get("argmax_agreement_poly_vs_gl") or meta.get("argmax_agreement"),
        "accuracy": meta.get("accuracy_gl_decrypted"),
        "logits_relative_l2": err.get("logits_relative_l2"),
    }


def main() -> None:
    rows = [
        row("gl_padded_pca32_single_n1", load("gl_padded_pca32_results.json")),
        row("gl_padded_pca32_packed_n32", load("gl_padded_pca32_packed_results_n32.json")),
        row("gl_padded_pca32_packed_n450", load("gl_padded_pca32_packed_results_n450.json")),
        row("gl_blocked_digits64_single_n1", load("gl_blocked_digits64_results.json")),
        row("gl_blocked_digits64_packed_n32", load("gl_blocked_digits64_packed_results_n32.json")),
        row("gl_blocked_digits64_packed_n450", load("gl_blocked_digits64_packed_results_n450.json")),
    ]
    singles = {r["condition"]: r for r in rows}
    for r in rows:
        if "padded_pca32_packed" in r["condition"]:
            base = singles["gl_padded_pca32_single_n1"]["runtime_per_sample_amortized_s"]
        elif "blocked_digits64_packed" in r["condition"]:
            base = singles["gl_blocked_digits64_single_n1"]["runtime_per_sample_amortized_s"]
        else:
            base = None
        if base and r["runtime_per_sample_amortized_s"]:
            r["per_sample_speedup_vs_single"] = base / r["runtime_per_sample_amortized_s"]
    write_csv(RESULTS / "gl_packing_summary.csv", rows)
    print(RESULTS / "gl_packing_summary.csv")


if __name__ == "__main__":
    main()
