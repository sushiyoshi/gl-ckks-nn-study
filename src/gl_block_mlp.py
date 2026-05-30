from __future__ import annotations

from typing import Any

import numpy as np

from .gl_packing import (
    broadcast_bias_to_columns,
    broadcast_weight_to_batches,
    packing_stats,
    split64_to_two_packed_tensors,
    unpack_logits,
)
from .logging_utils import ct_info, elapsed, exception_record, now_seconds
from .metrics import accuracy, argmax_agreement, error_metrics
from .model_pca32 import require_sample_count
from .model_raw64 import load_raw64_model
from .polynomial import fit_relu_power_polynomial


def block_structure() -> dict[str, int]:
    return {
        "linear1_input_blocks": 2,
        "linear1_weight_blocks": 2,
        "linear1_block_matmuls": 2,
        "linear2_block_matmuls": 1,
        "total_linear_block_matmuls": 3,
    }


def raw64_polynomial() -> tuple[float, np.ndarray]:
    model, arrays = load_raw64_model()
    z_train = model.relu_forward(arrays["x_train_64"])["z1"]
    radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z_train), 0.995)))))
    coeffs = fit_relu_power_polynomial(3, (-radius, radius)).astype(np.float64)
    return radius, coeffs


def classify_failure(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "level" in text or "depth" in text or "modulus" in text:
        return "level_insufficient"
    if "scale" in text:
        return "scale_mismatch"
    if "matrix_multiply" in text or "argument" in text or "incompatible" in text or "unsupported" in text:
        return "api_failure"
    return "api_failure"


def failure_payload(exc: BaseException, **context: Any) -> dict[str, Any]:
    payload = exception_record(exc, **context)
    payload["semantic_validation_passed"] = False
    payload["failure_category"] = classify_failure(exc)
    return payload


def encrypted_weight_raw64_mlp(n_samples: int) -> dict[str, Any]:
    from desilofhe import GLEngine

    t_total = now_seconds()
    model, arrays = load_raw64_model()
    require_sample_count(n_samples, arrays["x_test_64"].shape[0], label="x_test_64")
    n = n_samples
    x = np.asarray(arrays["x_test_64"][:n], dtype=np.float64)
    y = np.asarray(arrays["y_test"][:n])
    radius, coeffs = raw64_polynomial()
    logits_plain = model.poly_forward(x, coeffs)["logits"]

    engine = GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    t = now_seconds()
    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    had_key = engine.create_hadamard_multiplication_key(sk)
    key_generation_s = elapsed(t)

    x0_tensor, x1_tensor, layout = split64_to_two_packed_tensors(x, shape)
    used_batches = layout["used_batches"]
    w1_t = model.w1.T
    w1a_tensor = broadcast_weight_to_batches(w1_t[:, :32], shape, used_batches, 32, 32)
    w1b_tensor = broadcast_weight_to_batches(w1_t[:, 32:64], shape, used_batches, 32, 32)
    w2_tensor = broadcast_weight_to_batches(model.w2.T, shape, used_batches, 10, 32)
    b1_pt = engine.encode(broadcast_bias_to_columns(model.b1, shape, layout, 32))
    b2_pt = engine.encode(broadcast_bias_to_columns(model.b2, shape, layout, 10))

    timing: dict[str, float] = {}
    level_log: list[dict[str, Any]] = []
    t = now_seconds()
    ct_x0 = engine.encrypt(x0_tensor, sk)
    ct_x1 = engine.encrypt(x1_tensor, sk)
    timing["encryption_input_s"] = elapsed(t)
    level_log.append({"stage": "input_x0", **ct_info(ct_x0)})
    level_log.append({"stage": "input_x1", **ct_info(ct_x1)})

    t = now_seconds()
    ct_w1a = engine.encrypt(w1a_tensor, sk)
    ct_w1b = engine.encrypt(w1b_tensor, sk)
    ct_w2 = engine.encrypt(w2_tensor, sk)
    timing["encryption_weight_s"] = elapsed(t)
    level_log.append({"stage": "weight_w1a", **ct_info(ct_w1a)})
    level_log.append({"stage": "weight_w1b", **ct_info(ct_w1b)})
    level_log.append({"stage": "weight_w2", **ct_info(ct_w2)})

    t = now_seconds()
    h0 = engine.matrix_multiply(ct_w1a, ct_x0, mm_key)
    timing["linear1_block0_matrix_multiply_s"] = elapsed(t)
    level_log.append({"stage": "linear1_block0", **ct_info(h0)})

    t = now_seconds()
    h1 = engine.matrix_multiply(ct_w1b, ct_x1, mm_key)
    timing["linear1_block1_matrix_multiply_s"] = elapsed(t)
    level_log.append({"stage": "linear1_block1", **ct_info(h1)})

    t = now_seconds()
    ct_h = engine.add(h0, h1)
    ct_h = engine.add(ct_h, b1_pt)
    timing["linear1_accumulation_s"] = elapsed(t)
    level_log.append({"stage": "linear1_sum", **ct_info(ct_h)})

    t = now_seconds()
    ct_a = engine.evaluate_polynomial(ct_h, coeffs, had_key)
    timing["activation_s"] = elapsed(t)
    level_log.append({"stage": "activation", **ct_info(ct_a)})

    t = now_seconds()
    ct_y = engine.matrix_multiply(ct_w2, ct_a, mm_key)
    ct_y = engine.add(ct_y, b2_pt)
    timing["linear2_matrix_multiply_s"] = elapsed(t)
    level_log.append({"stage": "linear2", **ct_info(ct_y)})

    t = now_seconds()
    dec = np.asarray(engine.decrypt(ct_y, sk), dtype=np.float64)
    timing["decryption_s"] = elapsed(t)

    logits = unpack_logits(dec, layout, 10)
    err = error_metrics(logits_plain, logits, "logits")
    logits_allclose = bool(np.allclose(logits, logits_plain, rtol=1e-5, atol=1e-5))
    runtime_total = elapsed(t_total)
    server_only_s = sum(
        timing.get(key, 0.0)
        for key in (
            "linear1_block0_matrix_multiply_s",
            "linear1_block1_matrix_multiply_s",
            "linear1_accumulation_s",
            "activation_s",
            "linear2_matrix_multiply_s",
        )
    )
    total_minus_keygen_s = runtime_total - key_generation_s
    return {
        "ok": True,
        "semantic_validation_passed": bool(err["logits_relative_l2"] < 1e-5 and logits_allclose),
        "failure_category": None,
        "task_type": "raw64_mlp_poly_relu",
        "weight_privacy": "encrypted_weight",
        "activation": "degree3_poly",
        "polynomial_degree": 3,
        "polynomial_radius": radius,
        "coefficients": coeffs.tolist(),
        "logical_dims": [64, 32, 10],
        "shape": list(shape),
        "n_samples": n,
        "ciphertext_count": 5,
        "input_ciphertext_count": 2,
        "weight_ciphertext_count": 3,
        "block_matmul_count": 3,
        "block_structure": block_structure(),
        "used_batches": used_batches,
        "used_columns_last_batch": layout["used_columns_last_batch"],
        **packing_stats(n, 64, shape),
        "metadata": {
            "evaluation_sampling": "x_test_64/y_test are deterministic resamples of the base sklearn digits test split when n=8192; intended for throughput measurement, not unique-sample accuracy.",
            "baseline": "semantic validation compares against the degree3_poly MLP baseline, not the original ReLU MLP.",
            "model_path": "data/raw64_mlp.joblib",
        },
        **err,
        "logits_allclose": logits_allclose,
        "accuracy": accuracy(logits, y),
        "argmax_agreement": argmax_agreement(logits_plain, logits),
        "level_log": level_log,
        "timing_log": timing,
        "key_generation_s": key_generation_s,
        "server_only_s": server_only_s,
        "total_s": runtime_total,
        "runtime_total_s": runtime_total,
        "runtime_per_sample_s": runtime_total / n,
        "total_minus_keygen_s": total_minus_keygen_s,
        "server_per_sample_s": server_only_s / n,
        "no_keygen_per_sample_s": total_minus_keygen_s / n,
    }
