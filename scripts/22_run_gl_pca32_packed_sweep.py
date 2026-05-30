from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_packing import broadcast_bias_to_columns, broadcast_weight_to_batches, pack_columns, packing_stats, unpack_logits
from src.logging_utils import RESULTS, ct_info, elapsed, exception_record, now_seconds, write_csv
from src.metrics import accuracy, argmax_agreement, error_metrics
from src.model_pca32 import load_pca32_model, require_sample_count
from src.polynomial import fit_relu_power_polynomial


def flatten(n: int, meta: dict[str, Any]) -> dict[str, Any]:
    err = meta.get("error_metrics") or {}
    timing = meta.get("timing_log") or {}
    return {
        "condition": "gl_pca32_packed",
        "n_samples": n,
        "ok": meta.get("ok"),
        "semantic_validation_passed": meta.get("semantic_validation_passed"),
        "accuracy": meta.get("accuracy_gl_decrypted"),
        "argmax_agreement_poly_vs_fhe": meta.get("argmax_agreement_poly_vs_gl"),
        "relative_l2": err.get("logits_relative_l2"),
        "logits_linf": err.get("logits_linf"),
        "logits_mae": err.get("logits_mae"),
        "ciphertext_count": 1 if meta.get("ok") else None,
        "samples_per_ciphertext": meta.get("samples_per_ciphertext_capacity"),
        "slot_or_matrix_utilization": meta.get("matrix_entry_utilization_input"),
        "matrix_build_s": meta.get("matrix_build_s"),
        "key_generation_s": meta.get("key_generation_s"),
        "encryption_s_total": timing.get("encrypt_s"),
        "linear1_s_total": timing.get("linear1_add_s"),
        "activation_s_total": timing.get("activation_s"),
        "linear2_s_total": timing.get("linear2_add_s"),
        "decryption_s_total": timing.get("decrypt_s"),
        "runtime_total_s": meta.get("runtime_total_s"),
        "runtime_per_sample_s": meta.get("runtime_per_sample_amortized_s"),
        "exception": meta.get("exception"),
    }


def run_one(n: int, degree: int) -> dict[str, Any]:
    from desilofhe import GLEngine

    t_total = now_seconds()
    model, arrays = load_pca32_model()
    require_sample_count(n, arrays["x_test_32"].shape[0], label="x_test_32")
    x = arrays["x_test_32"][:n]
    y = arrays["y_test"][:n]
    z_train = model.relu_forward(arrays["x_train_32"])["z1"]
    radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z_train), 0.995)))))
    coeffs = fit_relu_power_polynomial(degree, (-radius, radius)).astype(np.float64)
    poly_plain = model.poly_forward(x, coeffs)["logits"]

    engine = GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    t = now_seconds()
    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    had_key = engine.create_hadamard_multiplication_key(sk)
    key_generation_s = elapsed(t)
    input_tensor, layout = pack_columns(x, shape)
    used_batches = layout["used_batches"]
    t = now_seconds()
    w1_tensor = broadcast_weight_to_batches(model.w1.T, shape, used_batches, 32, 32)
    w2_tensor = broadcast_weight_to_batches(model.w2.T, shape, used_batches, 10, 32)
    b1_tensor = broadcast_bias_to_columns(model.b1, shape, layout, 32)
    b2_tensor = broadcast_bias_to_columns(model.b2, shape, layout, 10)
    w1_pt = engine.encode(w1_tensor)
    w2_pt = engine.encode(w2_tensor)
    b1_pt = engine.encode(b1_tensor)
    b2_pt = engine.encode(b2_tensor)
    matrix_build_s = elapsed(t)

    timing = {}
    level_log = []
    t = now_seconds()
    ct = engine.encrypt(input_tensor, sk)
    timing["encrypt_s"] = elapsed(t)
    level_log.append({"stage": "input", **ct_info(ct)})
    t = now_seconds()
    ct = engine.matrix_multiply(w1_pt, ct, mm_key)
    ct = engine.add(ct, b1_pt)
    timing["linear1_add_s"] = elapsed(t)
    level_log.append({"stage": "linear1", **ct_info(ct)})
    t = now_seconds()
    ct = engine.evaluate_polynomial(ct, coeffs, had_key)
    timing["activation_s"] = elapsed(t)
    level_log.append({"stage": "activation", **ct_info(ct)})
    t = now_seconds()
    ct = engine.matrix_multiply(w2_pt, ct, mm_key)
    ct = engine.add(ct, b2_pt)
    timing["linear2_add_s"] = elapsed(t)
    level_log.append({"stage": "linear2", **ct_info(ct)})
    t = now_seconds()
    dec = np.asarray(engine.decrypt(ct, sk), dtype=np.float64)
    timing["decrypt_s"] = elapsed(t)
    logits = unpack_logits(dec, layout, 10)
    err = error_metrics(poly_plain, logits, "logits")
    runtime_total = elapsed(t_total)
    return {
        "ok": True,
        "semantic_validation_passed": bool(err["logits_relative_l2"] < 1e-5),
        "accuracy_gl_decrypted": accuracy(logits, y),
        "argmax_agreement_poly_vs_gl": argmax_agreement(poly_plain, logits),
        "error_metrics": err,
        "polynomial_degree": degree,
        "key_generation_s": key_generation_s,
        "matrix_build_s": matrix_build_s,
        "used_batches": used_batches,
        "used_columns_last_batch": layout["used_columns_last_batch"],
        **packing_stats(n, 32, shape),
        "operation_counts": {"matrix_multiply": 2, "add": 2, "polynomial": 1},
        "level_log": level_log,
        "timing_log": timing,
        "runtime_total_s": runtime_total,
        "runtime_per_sample_amortized_s": runtime_total / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--samples", nargs="*", type=int, default=[1, 32, 64, 128, 256, 450])
    args = parser.parse_args()
    rows = []
    for n in args.samples:
        try:
            meta = run_one(n, args.degree)
        except Exception as exc:
            meta = exception_record(exc, n_samples=n)
        rows.append(flatten(n, meta))
        write_csv(RESULTS / "gl_pca32_packed_sweep.csv", rows)
    print(RESULTS / "gl_pca32_packed_sweep.csv")


if __name__ == "__main__":
    main()
