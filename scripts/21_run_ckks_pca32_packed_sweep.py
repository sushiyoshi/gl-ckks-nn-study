from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ckks_packing import (
    dense_matrix_memory_stats,
    make_block_diagonal_linear_matrix,
    make_repeated_bias_vector,
    pack_samples_blocks,
    packing_stats,
    unpack_logits_blocks,
)
from src.logging_utils import RESULTS, ct_info, elapsed, exception_record, now_seconds, write_csv
from src.metrics import accuracy, argmax_agreement, error_metrics
from src.model_pca32 import load_pca32_model, require_sample_count
from src.polynomial import fit_relu_power_polynomial


def ckks_diagonal_indices(block_dim: int, slot_count: int) -> list[int]:
    offsets = list(range(-(block_dim - 1), block_dim))
    return sorted({offset % slot_count for offset in offsets})


def flatten(mode: str, n: int, meta: dict[str, Any]) -> dict[str, Any]:
    err = meta.get("error_metrics") or {}
    timing = meta.get("timing_log") or {}
    return {
        "condition": f"ckks_pca32_packed_{mode}",
        "mode": mode,
        "n_samples": n,
        "ok": meta.get("ok"),
        "semantic_validation_passed": meta.get("semantic_validation_passed"),
        "accuracy": meta.get("accuracy"),
        "argmax_agreement_poly_vs_fhe": meta.get("argmax_agreement_poly_vs_fhe"),
        "relative_l2": err.get("logits_relative_l2"),
        "logits_linf": err.get("logits_linf"),
        "logits_mae": err.get("logits_mae"),
        "ciphertext_count": meta.get("ciphertext_count"),
        "samples_per_ciphertext": meta.get("samples_per_ciphertext"),
        "slot_utilization": meta.get("slot_utilization"),
        "matrix_build_s": meta.get("matrix_build_s"),
        "matrix_encode_s": meta.get("matrix_encode_s"),
        "key_generation_s": timing.get("key_generation_s"),
        "encryption_s_total": timing.get("encryption_s_total"),
        "linear1_s_total": timing.get("linear1_s_total"),
        "activation_s_total": timing.get("activation_s_total"),
        "linear2_s_total": timing.get("linear2_s_total"),
        "decryption_s_total": timing.get("decryption_s_total"),
        "runtime_total_s": meta.get("runtime_total_s"),
        "runtime_per_sample_s": meta.get("runtime_per_sample_amortized_s"),
        "dense_matrix_mib_total": meta.get("dense_matrix_mib_total"),
        "exception": meta.get("exception"),
    }


def run_one(n: int, mode: str, degree: int, chunk_samples: int | None = None) -> dict[str, Any]:
    from desilofhe import Engine

    t_total = now_seconds()
    model, arrays = load_pca32_model()
    require_sample_count(n, arrays["x_test_32"].shape[0], label="x_test_32")
    x = arrays["x_test_32"][:n]
    y = arrays["y_test"][:n]
    z_train = model.relu_forward(arrays["x_train_32"])["z1"]
    radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z_train), 0.995)))))
    coeffs = fit_relu_power_polynomial(degree, (-radius, radius)).astype(np.float64)
    poly_plain = model.poly_forward(x, coeffs)["logits"]

    engine = Engine()
    slot_count = int(engine.slot_count)
    block_dim = 32
    chunks, layout = pack_samples_blocks(x, block_dim, slot_count, chunk_samples)
    samples_per_ct = layout["samples_per_ct"]
    timing = {"key_generation_s": 0.0, "encryption_s_total": 0.0, "linear1_s_total": 0.0, "activation_s_total": 0.0, "linear2_s_total": 0.0, "decryption_s_total": 0.0}
    t = now_seconds()
    sk = engine.create_secret_key()
    relin_key = engine.create_relinearization_key(sk)
    if mode == "dense":
        mat_key = engine.create_rotation_key(sk)
    else:
        mat_key = engine.create_matrix_multiplication_key(sk)
    timing["key_generation_s"] = elapsed(t)

    t = now_seconds()
    m1_dense = make_block_diagonal_linear_matrix(model.w1.T, block_dim, samples_per_ct, slot_count, 32, 32)
    m2_dense = make_block_diagonal_linear_matrix(model.w2.T, block_dim, samples_per_ct, slot_count, 10, 32)
    b1 = make_repeated_bias_vector(model.b1, block_dim, samples_per_ct, slot_count, 32)
    b2 = make_repeated_bias_vector(model.b2, block_dim, samples_per_ct, slot_count, 10)
    matrix_build_s = elapsed(t)
    mem = dense_matrix_memory_stats(m1_dense, m2_dense)
    matrix_encode_s = 0.0
    if mode == "dense":
        m1 = m1_dense
        m2 = m2_dense
    elif mode == "plainmatrix":
        t = now_seconds()
        m1 = engine.encode_to_plain_matrix(m1_dense)
        m2 = engine.encode_to_plain_matrix(m2_dense)
        matrix_encode_s = elapsed(t)
    elif mode == "lightplainmatrix":
        t = now_seconds()
        m1 = engine.encode_to_light_plain_matrix(m1_dense)
        m2 = engine.encode_to_light_plain_matrix(m2_dense)
        matrix_encode_s = elapsed(t)
    else:
        raise ValueError(f"unknown mode {mode}")

    decoded = []
    level_log = []
    for chunk_i, chunk in enumerate(chunks):
        t = now_seconds()
        ct = engine.encrypt(chunk, sk)
        timing["encryption_s_total"] += elapsed(t)
        if chunk_i == 0:
            level_log.append({"stage": "input", **ct_info(ct)})
        t = now_seconds()
        ct = engine.multiply_matrix(m1, ct, mat_key)
        ct = engine.add(ct, b1)
        timing["linear1_s_total"] += elapsed(t)
        if chunk_i == 0:
            level_log.append({"stage": "linear1", **ct_info(ct)})
        t = now_seconds()
        ct = engine.evaluate_polynomial(ct, [float(c) for c in coeffs], relin_key)
        timing["activation_s_total"] += elapsed(t)
        if chunk_i == 0:
            level_log.append({"stage": "activation", **ct_info(ct)})
        t = now_seconds()
        ct = engine.multiply_matrix(m2, ct, mat_key)
        ct = engine.add(ct, b2)
        timing["linear2_s_total"] += elapsed(t)
        if chunk_i == 0:
            level_log.append({"stage": "linear2", **ct_info(ct)})
        t = now_seconds()
        decoded.append(np.asarray(engine.decrypt(ct, sk), dtype=np.float64))
        timing["decryption_s_total"] += elapsed(t)

    logits = unpack_logits_blocks(decoded, layout, 10)
    err = error_metrics(poly_plain, logits, "logits")
    runtime_total = elapsed(t_total)
    return {
        "ok": True,
        "mode": mode,
        "semantic_validation_passed": bool(err["logits_relative_l2"] < 1e-5),
        "accuracy": accuracy(logits, y),
        "argmax_agreement_poly_vs_fhe": argmax_agreement(poly_plain, logits),
        "error_metrics": err,
        "polynomial_degree": degree,
        "matrix_build_s": matrix_build_s,
        "matrix_encode_s": matrix_encode_s,
        "diagonal_indices": None,
        **mem,
        **packing_stats(n, block_dim, slot_count, len(chunks)),
        "operation_counts": {"matrix_multiply": 2 * len(chunks), "add": 2 * len(chunks), "polynomial": len(chunks)},
        "level_log": level_log,
        "timing_log": timing,
        "runtime_total_s": runtime_total,
        "runtime_per_sample_amortized_s": runtime_total / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--samples", nargs="*", type=int, default=[1, 32, 64, 128, 256, 450])
    parser.add_argument("--mode", choices=["dense", "plainmatrix", "lightplainmatrix"], default="dense")
    parser.add_argument("--chunk-samples", type=int)
    args = parser.parse_args()

    rows = []
    for n in args.samples:
        try:
            meta = run_one(n, args.mode, args.degree, args.chunk_samples)
        except Exception as exc:
            meta = exception_record(exc, mode=args.mode, n_samples=n)
        rows.append(flatten(args.mode, n, meta))
        write_csv(RESULTS / f"ckks_pca32_packed_sweep_{args.mode}.csv", rows)
    if args.mode == "dense":
        write_csv(RESULTS / "ckks_pca32_packed_sweep.csv", rows)
    print(RESULTS / f"ckks_pca32_packed_sweep_{args.mode}.csv")


if __name__ == "__main__":
    main()
