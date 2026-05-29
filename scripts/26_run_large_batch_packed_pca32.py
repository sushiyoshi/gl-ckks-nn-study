from __future__ import annotations

import argparse
import json
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
    unpack_logits_blocks,
)
from src.gl_packing import broadcast_bias_to_columns, broadcast_weight_to_batches, pack_columns, unpack_logits
from src.logging_utils import RESULTS, ct_info, elapsed, exception_record, now_seconds, write_csv, write_json
from src.metrics import accuracy, argmax_agreement, error_metrics
from src.model_pca32 import load_pca32_model
from src.polynomial import fit_relu_power_polynomial


def sample_data(n: int, arrays: dict[str, np.ndarray], seed: int = 42) -> tuple[np.ndarray, np.ndarray, bool]:
    x_real = arrays["x_test_32"]
    y_real = arrays["y_test"]
    if n <= len(x_real):
        return x_real[:n], y_real[:n], False
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_real), size=n, replace=True)
    return x_real[idx], y_real[idx], True


def pack_coverage(n: int, ct_count: int, samples_per_ct: int) -> float:
    return n / (ct_count * samples_per_ct)


def run_gl(n: int, model, coeffs, arrays) -> dict[str, Any]:
    from desilofhe import GLEngine

    x, y, synthetic = sample_data(n, arrays)
    engine = GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    capacity = int(shape[0] * shape[2])
    sk = engine.create_secret_key()
    mmk = engine.create_matrix_multiplication_key(sk)
    had = engine.create_hadamard_multiplication_key(sk)
    input_tensor, layout = pack_columns(x, shape)
    used_batches = layout["used_batches"]
    w1 = engine.encode(broadcast_weight_to_batches(model.w1.T, shape, used_batches, 32, 32))
    w2 = engine.encode(broadcast_weight_to_batches(model.w2.T, shape, used_batches, 10, 32))
    b1 = engine.encode(broadcast_bias_to_columns(model.b1, shape, layout, 32))
    b2 = engine.encode(broadcast_bias_to_columns(model.b2, shape, layout, 10))
    t = now_seconds()
    ct = engine.encrypt(input_tensor, sk)
    encrypt_s = elapsed(t)
    t = now_seconds()
    ct = engine.matrix_multiply(w1, ct, mmk)
    ct = engine.add(ct, b1)
    linear1_s = elapsed(t)
    t = now_seconds()
    ct = engine.evaluate_polynomial(ct, coeffs, had)
    activation_s = elapsed(t)
    t = now_seconds()
    ct = engine.matrix_multiply(w2, ct, mmk)
    ct = engine.add(ct, b2)
    linear2_s = elapsed(t)
    t = now_seconds()
    dec = np.asarray(engine.decrypt(ct, sk), dtype=np.float64)
    decrypt_s = elapsed(t)
    logits = unpack_logits(dec, layout, 10)
    plain = model.poly_forward(x, coeffs)["logits"]
    err = error_metrics(plain, logits, "logits")
    runtime_total = encrypt_s + linear1_s + activation_s + linear2_s + decrypt_s
    return {
        "mode": "gl",
        "n_samples": n,
        "synthetic": synthetic,
        "semantic_validation_passed": bool(err["logits_relative_l2"] < 1e-5),
        "accuracy": accuracy(logits, y),
        "argmax_agreement_poly_vs_fhe": argmax_agreement(plain, logits),
        "error_metrics": err,
        "ciphertext_count": 1,
        "used_batches": used_batches,
        "used_columns_last_batch": layout["used_columns_last_batch"],
        "samples_per_ciphertext": capacity,
        "sample_utilization": n / capacity if capacity else 0.0,
        "slot_utilization": (n * x.shape[1]) / int(np.prod(shape)),
        "matrix_entry_utilization_input": (n * x.shape[1]) / int(np.prod(shape)),
        "runtime_total_s": runtime_total,
        "runtime_per_sample_s": runtime_total / n,
        "server_only_runtime_per_sample_s": (linear1_s + activation_s + linear2_s) / n,
        "offline": {},
        "online": {"encrypt_s_total": encrypt_s, "linear1_s_total": linear1_s, "activation_s_total": activation_s, "linear2_s_total": linear2_s, "decryption_s_total": decrypt_s},
        "dense_matrix_mib_total": None,
    }


def run_ckks(n: int, mode: str, model, coeffs, arrays) -> dict[str, Any]:
    from desilofhe import Engine

    if mode == "dense" and n >= 1024:
        return {"mode": mode, "n_samples": n, "skipped": True, "skip_reason": "CKKS dense skipped for n >= 1024 due to runtime", "semantic_validation_passed": None}
    x, y, synthetic = sample_data(n, arrays)
    engine = Engine()
    slots = int(engine.slot_count)
    block_dim = 32
    chunk = slots // block_dim
    chunks, layout = pack_samples_blocks(x, block_dim, slots, chunk)
    sk = engine.create_secret_key()
    rk = engine.create_rotation_key(sk)
    relk = engine.create_relinearization_key(sk)
    mmk = engine.create_matrix_multiplication_key(sk)
    m1_dense = make_block_diagonal_linear_matrix(model.w1.T, block_dim, chunk, slots, 32, 32)
    m2_dense = make_block_diagonal_linear_matrix(model.w2.T, block_dim, chunk, slots, 10, 32)
    if mode == "dense":
        m1, m2 = m1_dense, m2_dense
    elif mode == "plainmatrix":
        m1 = engine.encode_to_plain_matrix(m1_dense)
        m2 = engine.encode_to_plain_matrix(m2_dense)
    elif mode == "lightplainmatrix":
        m1 = engine.encode_to_light_plain_matrix(m1_dense)
        m2 = engine.encode_to_light_plain_matrix(m2_dense)
    else:
        raise ValueError(mode)
    b1 = make_repeated_bias_vector(model.b1, block_dim, chunk, slots, 32)
    b2 = make_repeated_bias_vector(model.b2, block_dim, chunk, slots, 10)
    encrypt_s = linear1_s = activation_s = linear2_s = decrypt_s = 0.0
    decoded = []
    for chunk_arr in chunks:
        t = now_seconds()
        ct = engine.encrypt(chunk_arr, sk)
        encrypt_s += elapsed(t)
        t = now_seconds()
        ct = engine.multiply_matrix(m1, ct, mmk if mode != "dense" else rk)
        ct = engine.add(ct, b1)
        linear1_s += elapsed(t)
        t = now_seconds()
        ct = engine.evaluate_polynomial(ct, coeffs, relk)
        activation_s += elapsed(t)
        t = now_seconds()
        ct = engine.multiply_matrix(m2, ct, mmk if mode != "dense" else rk)
        ct = engine.add(ct, b2)
        linear2_s += elapsed(t)
        t = now_seconds()
        decoded.append(np.asarray(engine.decrypt(ct, sk), dtype=np.float64))
        decrypt_s += elapsed(t)
    logits = unpack_logits_blocks(decoded, layout, 10)
    plain = model.poly_forward(x, coeffs)["logits"]
    err = error_metrics(plain, logits, "logits")
    runtime_total = encrypt_s + linear1_s + activation_s + linear2_s + decrypt_s
    return {
        "mode": mode,
        "n_samples": n,
        "synthetic": synthetic,
        "semantic_validation_passed": bool(err["logits_relative_l2"] < 1e-5),
        "accuracy": accuracy(logits, y),
        "argmax_agreement_poly_vs_fhe": argmax_agreement(plain, logits),
        "error_metrics": err,
        "ciphertext_count": len(chunks),
        "samples_per_ciphertext": chunk,
        "sample_utilization": n / (len(chunks) * chunk),
        "slot_utilization": (n * block_dim) / (len(chunks) * slots),
        "runtime_total_s": runtime_total,
        "runtime_per_sample_s": runtime_total / n,
        "server_only_runtime_per_sample_s": (linear1_s + activation_s + linear2_s) / n,
        "offline": {},
        "online": {"encrypt_s_total": encrypt_s, "linear1_s_total": linear1_s, "activation_s_total": activation_s, "linear2_s_total": linear2_s, "decryption_s_total": decrypt_s},
        "dense_matrix_mib_total": dense_matrix_memory_stats(m1_dense, m2_dense)["dense_matrix_mib_total"],
    }


def row_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": result.get("mode"),
        "n_samples": result.get("n_samples"),
        "synthetic": result.get("synthetic"),
        "semantic_validation_passed": result.get("semantic_validation_passed"),
        "accuracy": result.get("accuracy"),
        "argmax_agreement_poly_vs_fhe": result.get("argmax_agreement_poly_vs_fhe"),
        "relative_l2": (result.get("error_metrics") or {}).get("logits_relative_l2"),
        "ciphertext_count": result.get("ciphertext_count"),
        "samples_per_ciphertext": result.get("samples_per_ciphertext"),
        "sample_utilization": result.get("sample_utilization"),
        "slot_utilization": result.get("slot_utilization"),
        "runtime_total_s": result.get("runtime_total_s"),
        "runtime_per_sample_s": result.get("runtime_per_sample_s"),
        "server_only_runtime_per_sample_s": result.get("server_only_runtime_per_sample_s"),
        "skip_reason": result.get("skip_reason"),
        "dense_matrix_mib_total": result.get("dense_matrix_mib_total"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--samples", nargs="*", type=int, default=[450, 1024, 2048, 4096, 8192])
    parser.add_argument("--modes", nargs="*", default=["gl", "lightplainmatrix", "plainmatrix", "dense"])
    args = parser.parse_args()

    model, arrays = load_pca32_model()
    z_train = model.relu_forward(arrays["x_train_32"])["z1"]
    radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z_train), 0.995)))))
    coeffs = fit_relu_power_polynomial(args.degree, (-radius, radius)).astype(np.float64)
    rows = []
    payload = {"modes": args.modes, "samples": args.samples, "results": []}
    for n in args.samples:
        for mode in args.modes:
            try:
                if mode == "gl":
                    res = run_gl(n, model, coeffs, arrays)
                else:
                    res = run_ckks(n, mode, model, coeffs, arrays)
            except Exception as exc:
                res = exception_record(exc, mode=mode, n_samples=n)
                res["mode"] = mode
                res["n_samples"] = n
                res["synthetic"] = n > len(arrays["x_test_32"])
            rows.append(row_from_result(res))
            payload["results"].append(res)
            write_csv(RESULTS / "large_batch_packed_pca32.csv", rows)
    write_json(RESULTS / "large_batch_packed_pca32.json", payload)
    print(RESULTS / "large_batch_packed_pca32.csv")


if __name__ == "__main__":
    main()
