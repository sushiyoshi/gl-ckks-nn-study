from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ckks_packing import dense_matrix_memory_stats, make_block_diagonal_linear_matrix, make_repeated_bias_vector, pack_samples_blocks, unpack_logits_blocks
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


def ckks_dense_run(engine, x: np.ndarray, y: np.ndarray, coeffs: np.ndarray, model, mode: str) -> dict[str, Any]:
    slots = int(engine.slot_count)
    t = now_seconds()
    sk = engine.create_secret_key()
    key_generation_s = elapsed(t)
    t = now_seconds()
    rk = engine.create_rotation_key(sk)
    relk = engine.create_relinearization_key(sk)
    mmk = engine.create_matrix_multiplication_key(sk)
    eval_key_generation_s = elapsed(t)
    block_dim = 32
    chunk = slots // block_dim
    t = now_seconds()
    chunks, layout = pack_samples_blocks(x, block_dim, slots, chunk)
    input_tensor_construction_s = elapsed(t)
    t = now_seconds()
    m1 = make_block_diagonal_linear_matrix(model.w1.T, block_dim, chunk, slots, 32, 32)
    m2 = make_block_diagonal_linear_matrix(model.w2.T, block_dim, chunk, slots, 10, 32)
    matrix_build_s = elapsed(t)
    t = now_seconds()
    b1 = make_repeated_bias_vector(model.b1, block_dim, chunk, slots, 32)
    b2 = make_repeated_bias_vector(model.b2, block_dim, chunk, slots, 10)
    bias_construction_s = elapsed(t)
    online = {"encryption_s_total": 0.0, "linear1_s_total": 0.0, "activation_s_total": 0.0, "linear2_s_total": 0.0, "decryption_s_total": 0.0}
    decoded = []
    for chunk_arr in chunks:
        t = now_seconds()
        ct = engine.encrypt(chunk_arr, sk)
        online["encryption_s_total"] += elapsed(t)
        t = now_seconds()
        ct = engine.multiply_matrix(m1, ct, rk)
        ct = engine.add(ct, b1)
        online["linear1_s_total"] += elapsed(t)
        t = now_seconds()
        ct = engine.evaluate_polynomial(ct, [float(c) for c in coeffs], relk)
        online["activation_s_total"] += elapsed(t)
        t = now_seconds()
        ct = engine.multiply_matrix(m2, ct, rk)
        ct = engine.add(ct, b2)
        online["linear2_s_total"] += elapsed(t)
        t = now_seconds()
        decoded.append(np.asarray(engine.decrypt(ct, sk), dtype=np.float64))
        online["decryption_s_total"] += elapsed(t)
    logits = unpack_logits_blocks(decoded, layout, 10)
    return {
        "mode": mode,
        "n_samples": len(x),
        "slots": slots,
        "ciphertext_count": len(chunks),
        "samples_per_ciphertext": chunk,
        "sample_utilization": len(x) / (len(chunks) * chunk),
        "semantic_validation_passed": bool(error_metrics(model.poly_forward(x, coeffs)["logits"], logits, "logits")["logits_relative_l2"] < 1e-5),
        "accuracy": accuracy(logits, y),
        "argmax_agreement_poly_vs_fhe": argmax_agreement(model.poly_forward(x, coeffs)["logits"], logits),
        "error_metrics": error_metrics(model.poly_forward(x, coeffs)["logits"], logits, "logits"),
        "offline": {
            "key_generation_s": key_generation_s,
            "evaluation_key_generation_s": eval_key_generation_s,
            "input_tensor_construction_s": input_tensor_construction_s,
            "matrix_construction_s": matrix_build_s,
            "bias_construction_s": bias_construction_s,
        },
        "online": online,
        "server_only_online_s": online["linear1_s_total"] + online["activation_s_total"] + online["linear2_s_total"],
        "runtime_total_s": sum(online.values()),
        "runtime_per_sample_s": sum(online.values()) / len(x),
        "server_only_runtime_per_sample_s": (online["linear1_s_total"] + online["activation_s_total"] + online["linear2_s_total"]) / len(x),
        "dense_matrix_mib_total": dense_matrix_memory_stats(m1, m2)["dense_matrix_mib_total"],
    }


def ckks_plain_run(engine, x: np.ndarray, y: np.ndarray, coeffs: np.ndarray, model, mode: str) -> dict[str, Any]:
    slots = int(engine.slot_count)
    t = now_seconds()
    sk = engine.create_secret_key()
    key_generation_s = elapsed(t)
    t = now_seconds()
    rk = engine.create_rotation_key(sk)
    relk = engine.create_relinearization_key(sk)
    mmk = engine.create_matrix_multiplication_key(sk)
    eval_key_generation_s = elapsed(t)
    block_dim = 32
    chunk = slots // block_dim
    t = now_seconds()
    chunks, layout = pack_samples_blocks(x, block_dim, slots, chunk)
    input_tensor_construction_s = elapsed(t)
    t = now_seconds()
    m1_dense = make_block_diagonal_linear_matrix(model.w1.T, block_dim, chunk, slots, 32, 32)
    m2_dense = make_block_diagonal_linear_matrix(model.w2.T, block_dim, chunk, slots, 10, 32)
    matrix_build_s = elapsed(t)
    t = now_seconds()
    if mode == "plainmatrix":
        m1 = engine.encode_to_plain_matrix(m1_dense)
        m2 = engine.encode_to_plain_matrix(m2_dense)
    elif mode == "lightplainmatrix":
        m1 = engine.encode_to_light_plain_matrix(m1_dense)
        m2 = engine.encode_to_light_plain_matrix(m2_dense)
    else:
        raise ValueError(mode)
    matrix_encode_s = elapsed(t)
    t = now_seconds()
    b1 = make_repeated_bias_vector(model.b1, block_dim, chunk, slots, 32)
    b2 = make_repeated_bias_vector(model.b2, block_dim, chunk, slots, 10)
    bias_construction_s = elapsed(t)
    online = {"encryption_s_total": 0.0, "linear1_s_total": 0.0, "activation_s_total": 0.0, "linear2_s_total": 0.0, "decryption_s_total": 0.0}
    decoded = []
    for chunk_arr in chunks:
        t = now_seconds()
        ct = engine.encrypt(chunk_arr, sk)
        online["encryption_s_total"] += elapsed(t)
        t = now_seconds()
        ct = engine.multiply_matrix(m1, ct, mmk)
        ct = engine.add(ct, b1)
        online["linear1_s_total"] += elapsed(t)
        t = now_seconds()
        ct = engine.evaluate_polynomial(ct, [float(c) for c in coeffs], relk)
        online["activation_s_total"] += elapsed(t)
        t = now_seconds()
        ct = engine.multiply_matrix(m2, ct, mmk)
        ct = engine.add(ct, b2)
        online["linear2_s_total"] += elapsed(t)
        t = now_seconds()
        decoded.append(np.asarray(engine.decrypt(ct, sk), dtype=np.float64))
        online["decryption_s_total"] += elapsed(t)
    logits = unpack_logits_blocks(decoded, layout, 10)
    return {
        "mode": mode,
        "n_samples": len(x),
        "slots": slots,
        "ciphertext_count": len(chunks),
        "samples_per_ciphertext": chunk,
        "sample_utilization": len(x) / (len(chunks) * chunk),
        "semantic_validation_passed": bool(error_metrics(model.poly_forward(x, coeffs)["logits"], logits, "logits")["logits_relative_l2"] < 1e-5),
        "accuracy": accuracy(logits, y),
        "argmax_agreement_poly_vs_fhe": argmax_agreement(model.poly_forward(x, coeffs)["logits"], logits),
        "error_metrics": error_metrics(model.poly_forward(x, coeffs)["logits"], logits, "logits"),
        "offline": {
            "key_generation_s": key_generation_s,
            "evaluation_key_generation_s": eval_key_generation_s,
            "input_tensor_construction_s": input_tensor_construction_s,
            "matrix_encode_s": matrix_encode_s,
            "matrix_construction_s": matrix_build_s,
            "bias_construction_s": bias_construction_s,
        },
        "online": online,
        "server_only_online_s": online["linear1_s_total"] + online["activation_s_total"] + online["linear2_s_total"],
        "runtime_total_s": sum(online.values()),
        "runtime_per_sample_s": sum(online.values()) / len(x),
        "server_only_runtime_per_sample_s": (online["linear1_s_total"] + online["activation_s_total"] + online["linear2_s_total"]) / len(x),
        "dense_matrix_mib_total": dense_matrix_memory_stats(m1_dense, m2_dense)["dense_matrix_mib_total"],
    }


def gl_run(engine, x: np.ndarray, y: np.ndarray, coeffs: np.ndarray, model) -> dict[str, Any]:
    shape = tuple(int(v) for v in engine.shape)
    t = now_seconds()
    sk = engine.create_secret_key()
    key_generation_s = elapsed(t)
    t = now_seconds()
    mmk = engine.create_matrix_multiplication_key(sk)
    had = engine.create_hadamard_multiplication_key(sk)
    eval_key_generation_s = elapsed(t)
    t = now_seconds()
    input_tensor, layout = pack_columns(x, shape)
    input_tensor_construction_s = elapsed(t)
    used_batches = layout["used_batches"]
    t = now_seconds()
    w1 = engine.encode(broadcast_weight_to_batches(model.w1.T, shape, used_batches, 32, 32))
    w2 = engine.encode(broadcast_weight_to_batches(model.w2.T, shape, used_batches, 10, 32))
    weight_encoding_s = elapsed(t)
    t = now_seconds()
    b1 = engine.encode(broadcast_bias_to_columns(model.b1, shape, layout, 32))
    b2 = engine.encode(broadcast_bias_to_columns(model.b2, shape, layout, 10))
    bias_construction_s = elapsed(t)
    offline = {
        "key_generation_s": key_generation_s,
        "evaluation_key_generation_s": eval_key_generation_s,
        "input_tensor_construction_s": input_tensor_construction_s,
        "weight_encoding_s": weight_encoding_s,
        "matrix_construction_s": 0.0,
        "bias_construction_s": bias_construction_s,
    }
    online = {"encryption_s_total": 0.0, "linear1_s_total": 0.0, "activation_s_total": 0.0, "linear2_s_total": 0.0, "decryption_s_total": 0.0}
    t = now_seconds()
    ct = engine.encrypt(input_tensor, sk)
    online["encryption_s_total"] = elapsed(t)
    t = now_seconds()
    ct = engine.matrix_multiply(w1, ct, mmk)
    ct = engine.add(ct, b1)
    online["linear1_s_total"] = elapsed(t)
    t = now_seconds()
    ct = engine.evaluate_polynomial(ct, coeffs, had)
    online["activation_s_total"] = elapsed(t)
    t = now_seconds()
    ct = engine.matrix_multiply(w2, ct, mmk)
    ct = engine.add(ct, b2)
    online["linear2_s_total"] = elapsed(t)
    t = now_seconds()
    dec = np.asarray(engine.decrypt(ct, sk), dtype=np.float64)
    online["decryption_s_total"] = elapsed(t)
    logits = unpack_logits(dec, layout, 10)
    return {
        "mode": "packed",
        "n_samples": len(x),
        "slot_count": int(np.prod(shape)),
        "ciphertext_count": 1,
        "samples_per_ciphertext": int(shape[0] * shape[2]),
        "used_batches": used_batches,
        "used_columns_last_batch": layout["used_columns_last_batch"],
        "sample_utilization": len(x) / (shape[0] * shape[2]),
        "matrix_entry_utilization_input": len(x) * x.shape[1] / int(np.prod(shape)),
        "semantic_validation_passed": bool(error_metrics(model.poly_forward(x, coeffs)["logits"], logits, "logits")["logits_relative_l2"] < 1e-5),
        "accuracy": accuracy(logits, y),
        "argmax_agreement_poly_vs_fhe": argmax_agreement(model.poly_forward(x, coeffs)["logits"], logits),
        "error_metrics": error_metrics(model.poly_forward(x, coeffs)["logits"], logits, "logits"),
        "offline": offline,
        "online": online,
        "server_only_online_s": online["linear1_s_total"] + online["activation_s_total"] + online["linear2_s_total"],
        "runtime_total_s": sum(online.values()),
        "runtime_per_sample_s": sum(online.values()) / len(x),
        "server_only_runtime_per_sample_s": (online["linear1_s_total"] + online["activation_s_total"] + online["linear2_s_total"]) / len(x),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=450)
    parser.add_argument("--modes", nargs="*", default=["dense", "plainmatrix", "lightplainmatrix", "gl"])
    args = parser.parse_args()

    model, arrays = load_pca32_model()
    x, y, synthetic = sample_data(args.n_samples, arrays)
    z_train = model.relu_forward(arrays["x_train_32"])["z1"]
    radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z_train), 0.995)))))
    coeffs = fit_relu_power_polynomial(args.degree, (-radius, radius)).astype(np.float64)
    rows = []
    payload = {"n_samples": int(args.n_samples), "synthetic": synthetic, "modes": args.modes, "results": []}

    for mode in args.modes:
        try:
            if mode == "gl":
                from desilofhe import GLEngine
                engine = GLEngine()
                result = gl_run(engine, x, y, coeffs, model)
            else:
                from desilofhe import Engine
                engine = Engine()
                if mode == "dense":
                    result = ckks_dense_run(engine, x, y, coeffs, model, mode)
                else:
                    result = ckks_plain_run(engine, x, y, coeffs, model, mode)
            result["mode"] = mode
            result["n_samples"] = len(x)
            result["synthetic"] = synthetic
            result["offline_total_s"] = sum(v for v in result.get("offline", {}).values() if isinstance(v, (int, float)))
            result["online_total_s"] = sum(v for v in result.get("online", {}).values() if isinstance(v, (int, float)))
            rows.append({
                "mode": mode,
                "n_samples": len(x),
                "synthetic": synthetic,
                "semantic_validation_passed": result.get("semantic_validation_passed"),
                "accuracy": result.get("accuracy"),
                "argmax_agreement_poly_vs_fhe": result.get("argmax_agreement_poly_vs_fhe"),
                "offline_total_s": result["offline_total_s"],
                "online_total_s": result["online_total_s"],
                "server_only_online_s": result.get("server_only_online_s"),
                "runtime_total_s": result.get("runtime_total_s"),
                "runtime_per_sample_s": result.get("runtime_per_sample_s"),
                "server_only_runtime_per_sample_s": result.get("server_only_runtime_per_sample_s"),
                "ciphertext_count": result.get("ciphertext_count"),
                "sample_utilization": result.get("sample_utilization"),
                "dense_matrix_mib_total": result.get("dense_matrix_mib_total"),
                "error_metrics": json.dumps(result.get("error_metrics"), default=float),
                "offline": json.dumps(result.get("offline"), default=float),
                "online": json.dumps(result.get("online"), default=float),
            })
            payload["results"].append(result)
        except Exception as exc:
            rec = exception_record(exc, mode=mode, n_samples=len(x), synthetic=synthetic)
            rows.append({"mode": mode, "n_samples": len(x), "synthetic": synthetic, "semantic_validation_passed": False, "exception": rec["exception"], "offline_total_s": None, "online_total_s": None, "server_only_online_s": None, "runtime_total_s": None, "runtime_per_sample_s": None})
            payload["results"].append(rec)
    write_json(RESULTS / "steady_state_packed_pca32.json", payload)
    write_csv(RESULTS / "steady_state_packed_pca32.csv", rows)
    print(RESULTS / "steady_state_packed_pca32.csv")


if __name__ == "__main__":
    main()
