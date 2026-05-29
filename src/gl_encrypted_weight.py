from __future__ import annotations

from typing import Any

import numpy as np

from .gl_packing import (
    broadcast_bias_to_columns,
    broadcast_weight_to_batches,
    pack_columns,
    packing_stats,
    unpack_logits,
)
from .logging_utils import ct_info, elapsed, exception_record, now_seconds
from .metrics import accuracy, argmax_agreement, error_metrics
from .model_pca32 import load_pca32_model
from .polynomial import eval_power_polynomial, fit_relu_power_polynomial


def classify_failure(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "level" in text or "depth" in text or "modulus" in text:
        return "level_insufficient"
    if "scale" in text:
        return "scale_mismatch"
    if "matrix_multiply" in text or "argument" in text or "incompatible" in text or "unsupported" in text:
        return "api_failure"
    return "api_failure"


def scalar_metrics(reference: np.ndarray, actual: np.ndarray, prefix: str) -> dict[str, Any]:
    err = error_metrics(reference, actual, prefix)
    return {
        f"{prefix}_relative_l2": err[f"{prefix}_relative_l2"],
        f"{prefix}_linf": err[f"{prefix}_linf"],
        f"{prefix}_mae": err[f"{prefix}_mae"],
        f"{prefix}_allclose": bool(np.allclose(actual, reference, rtol=1e-5, atol=1e-5)),
    }


def operand_kind(obj: Any) -> str:
    if obj is None:
        return "none"
    kind = type(obj).__name__
    if "Cipher" in kind or kind.startswith("Ct"):
        return "ciphertext"
    if "Plain" in kind:
        return "plaintext"
    if kind == "ndarray":
        return "numpy.ndarray"
    return kind


def operand_types(ct_w1: Any, ct_w2: Any, ct_x: Any) -> dict[str, Any]:
    return {
        "ct_W1_type": type(ct_w1).__name__ if ct_w1 is not None else None,
        "ct_W2_type": type(ct_w2).__name__ if ct_w2 is not None else None,
        "ct_X_type": type(ct_x).__name__ if ct_x is not None else None,
        "matrix_multiply_operand_kinds": {
            "left": operand_kind(ct_w1),
            "right": operand_kind(ct_x),
            "left_w2": operand_kind(ct_w2),
        },
    }


def encrypted_weight_linear(n_samples: int) -> dict[str, Any]:
    from desilofhe import GLEngine

    t_total = now_seconds()
    model, arrays = load_pca32_model()
    n = min(n_samples, arrays["x_test_32"].shape[0])
    x = arrays["x_test_32"][:n]
    z_plain = model.relu_forward(x)["z1"]

    engine = GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    t = now_seconds()
    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    key_generation_s = elapsed(t)

    input_tensor, layout = pack_columns(x, shape)
    used_batches = layout["used_batches"]
    w1_tensor = broadcast_weight_to_batches(model.w1.T, shape, used_batches, 32, 32)
    b1_tensor = broadcast_bias_to_columns(model.b1, shape, layout, 32)
    b1_pt = engine.encode(b1_tensor)

    timing: dict[str, float] = {}
    level_log: list[dict[str, Any]] = []
    t = now_seconds()
    ct_x = engine.encrypt(input_tensor, sk)
    timing["encryption_input_s"] = elapsed(t)
    level_log.append({"stage": "input", **ct_info(ct_x)})
    t = now_seconds()
    ct_w1 = engine.encrypt(w1_tensor, sk)
    timing["encryption_weight_s"] = elapsed(t)
    level_log.append({"stage": "weight_w1", **ct_info(ct_w1)})
    t = now_seconds()
    ct_h = engine.matrix_multiply(ct_w1, ct_x, mm_key)
    ct_h = engine.add(ct_h, b1_pt)
    timing["matrix_multiply_s"] = elapsed(t)
    level_log.append({"stage": "linear1", **ct_info(ct_h)})
    t = now_seconds()
    dec = np.asarray(engine.decrypt(ct_h, sk), dtype=np.float64)
    timing["decryption_s"] = elapsed(t)

    hidden = unpack_logits(dec, layout, 32)
    metrics = scalar_metrics(z_plain, hidden, "hidden")
    runtime_total = elapsed(t_total)
    return {
        "ok": True,
        "semantic_validation_passed": bool(metrics["hidden_relative_l2"] < 1e-5 and metrics["hidden_allclose"]),
        "failure_category": None,
        "task_type": "linear",
        "weight_privacy": "encrypted_weight",
        "activation": "none",
        "shape": shape,
        "logical_dims": [32, 32],
        "n_samples": n,
        "ciphertext_count": 2,
        "key_generation_s": key_generation_s,
        "input_encryption_s": timing.get("encryption_input_s"),
        "weight_encryption_s": timing.get("encryption_weight_s"),
        "used_batches": used_batches,
        "used_columns_last_batch": layout["used_columns_last_batch"],
        **packing_stats(n, 32, shape),
        **operand_types(ct_w1, None, ct_x),
        **metrics,
        "level_log": level_log,
        "timing_log": timing,
        "server_only_s": timing.get("matrix_multiply_s"),
        "total_s": runtime_total,
        "runtime_total_s": runtime_total,
        "runtime_per_sample_s": runtime_total / n,
    }


def encrypted_weight_two_linear(n_samples: int) -> dict[str, Any]:
    from desilofhe import GLEngine

    t_total = now_seconds()
    model, arrays = load_pca32_model()
    n = min(n_samples, arrays["x_test_32"].shape[0])
    x = arrays["x_test_32"][:n]
    y = arrays["y_test"][:n]
    hidden_plain = x @ model.w1 + model.b1
    logits_plain = hidden_plain @ model.w2 + model.b2

    engine = GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    t = now_seconds()
    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    key_generation_s = elapsed(t)

    input_tensor, layout = pack_columns(x, shape)
    used_batches = layout["used_batches"]
    w1_tensor = broadcast_weight_to_batches(model.w1.T, shape, used_batches, 32, 32)
    w2_tensor = broadcast_weight_to_batches(model.w2.T, shape, used_batches, 10, 32)
    b1_pt = engine.encode(broadcast_bias_to_columns(model.b1, shape, layout, 32))
    b2_pt = engine.encode(broadcast_bias_to_columns(model.b2, shape, layout, 10))

    timing: dict[str, float] = {}
    level_log: list[dict[str, Any]] = []
    t = now_seconds()
    ct_x = engine.encrypt(input_tensor, sk)
    timing["encryption_input_s"] = elapsed(t)
    level_log.append({"stage": "input", **ct_info(ct_x)})
    t = now_seconds()
    ct_w1 = engine.encrypt(w1_tensor, sk)
    ct_w2 = engine.encrypt(w2_tensor, sk)
    timing["encryption_weight_s"] = elapsed(t)
    level_log.append({"stage": "weight_w1", **ct_info(ct_w1)})
    level_log.append({"stage": "weight_w2", **ct_info(ct_w2)})
    t = now_seconds()
    ct_h = engine.matrix_multiply(ct_w1, ct_x, mm_key)
    ct_h = engine.add(ct_h, b1_pt)
    timing["linear1_matrix_multiply_s"] = elapsed(t)
    level_log.append({"stage": "linear1", **ct_info(ct_h)})
    t = now_seconds()
    ct_y = engine.matrix_multiply(ct_w2, ct_h, mm_key)
    ct_y = engine.add(ct_y, b2_pt)
    timing["linear2_matrix_multiply_s"] = elapsed(t)
    level_log.append({"stage": "linear2", **ct_info(ct_y)})
    t = now_seconds()
    dec = np.asarray(engine.decrypt(ct_y, sk), dtype=np.float64)
    timing["decryption_s"] = elapsed(t)

    logits = unpack_logits(dec, layout, 10)
    metrics = scalar_metrics(logits_plain, logits, "logits")
    runtime_total = elapsed(t_total)
    return {
        "ok": True,
        "semantic_validation_passed": bool(metrics["logits_relative_l2"] < 1e-5 and metrics["logits_allclose"]),
        "failure_category": None,
        "task_type": "two_linear",
        "weight_privacy": "encrypted_weight",
        "activation": "none",
        "shape": shape,
        "logical_dims": [32, 32, 10],
        "n_samples": n,
        "ciphertext_count": 3,
        "key_generation_s": key_generation_s,
        "input_encryption_s": timing.get("encryption_input_s"),
        "weight_encryption_s": timing.get("encryption_weight_s"),
        "used_batches": used_batches,
        "used_columns_last_batch": layout["used_columns_last_batch"],
        **packing_stats(n, 32, shape),
        **operand_types(ct_w1, ct_w2, ct_x),
        **metrics,
        "accuracy": accuracy(logits, y),
        "argmax_agreement": argmax_agreement(logits_plain, logits),
        "level_log": level_log,
        "timing_log": timing,
        "server_only_s": (timing.get("linear1_matrix_multiply_s") or 0.0) + (timing.get("linear2_matrix_multiply_s") or 0.0),
        "total_s": runtime_total,
        "runtime_total_s": runtime_total,
        "runtime_per_sample_s": runtime_total / n,
    }


def encrypted_weight_mlp(n_samples: int, degree: int) -> dict[str, Any]:
    from desilofhe import GLEngine

    t_total = now_seconds()
    model, arrays = load_pca32_model()
    n = min(n_samples, arrays["x_test_32"].shape[0])
    x = arrays["x_test_32"][:n]
    y = arrays["y_test"][:n]
    z_train = model.relu_forward(arrays["x_train_32"])["z1"]
    radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z_train), 0.995)))))
    coeffs = fit_relu_power_polynomial(degree, (-radius, radius)).astype(np.float64)
    logits_plain = model.poly_forward(x, coeffs)["logits"]

    engine = GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    t = now_seconds()
    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    had_key = engine.create_hadamard_multiplication_key(sk)
    key_generation_s = elapsed(t)

    input_tensor, layout = pack_columns(x, shape)
    used_batches = layout["used_batches"]
    w1_tensor = broadcast_weight_to_batches(model.w1.T, shape, used_batches, 32, 32)
    w2_tensor = broadcast_weight_to_batches(model.w2.T, shape, used_batches, 10, 32)
    b1_pt = engine.encode(broadcast_bias_to_columns(model.b1, shape, layout, 32))
    b2_pt = engine.encode(broadcast_bias_to_columns(model.b2, shape, layout, 10))

    timing: dict[str, float] = {}
    level_log: list[dict[str, Any]] = []
    t = now_seconds()
    ct_x = engine.encrypt(input_tensor, sk)
    timing["encryption_input_s"] = elapsed(t)
    level_log.append({"stage": "input", **ct_info(ct_x)})
    t = now_seconds()
    ct_w1 = engine.encrypt(w1_tensor, sk)
    ct_w2 = engine.encrypt(w2_tensor, sk)
    timing["encryption_weight_s"] = elapsed(t)
    level_log.append({"stage": "weight_w1", **ct_info(ct_w1)})
    level_log.append({"stage": "weight_w2", **ct_info(ct_w2)})
    t = now_seconds()
    ct_h = engine.matrix_multiply(ct_w1, ct_x, mm_key)
    ct_h = engine.add(ct_h, b1_pt)
    timing["linear1_matrix_multiply_s"] = elapsed(t)
    level_log.append({"stage": "linear1", **ct_info(ct_h)})
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
    metrics = scalar_metrics(logits_plain, logits, "logits")
    runtime_total = elapsed(t_total)
    return {
        "ok": True,
        "semantic_validation_passed": bool(metrics["logits_relative_l2"] < 1e-5 and metrics["logits_allclose"]),
        "failure_category": None,
        "task_type": "mlp_poly_relu",
        "weight_privacy": "encrypted_weight",
        "activation": f"degree{degree}_poly",
        "polynomial_degree": degree,
        "polynomial_radius": radius,
        "shape": shape,
        "logical_dims": [32, 32, 10],
        "n_samples": n,
        "ciphertext_count": 3,
        "key_generation_s": key_generation_s,
        "input_encryption_s": timing.get("encryption_input_s"),
        "weight_encryption_s": timing.get("encryption_weight_s"),
        "used_batches": used_batches,
        "used_columns_last_batch": layout["used_columns_last_batch"],
        **packing_stats(n, 32, shape),
        **operand_types(ct_w1, ct_w2, ct_x),
        **metrics,
        "accuracy": accuracy(logits, y),
        "argmax_agreement": argmax_agreement(logits_plain, logits),
        "level_log": level_log,
        "timing_log": timing,
        "server_only_s": (timing.get("linear1_matrix_multiply_s") or 0.0) + (timing.get("activation_s") or 0.0) + (timing.get("linear2_matrix_multiply_s") or 0.0),
        "total_s": runtime_total,
        "runtime_total_s": runtime_total,
        "runtime_per_sample_s": runtime_total / n,
    }


def failure_payload(exc: BaseException, **context: Any) -> dict[str, Any]:
    payload = exception_record(exc, **context)
    payload["semantic_validation_passed"] = False
    payload["failure_category"] = classify_failure(exc)
    return payload


def packed_plaintext_logits(
    x: np.ndarray,
    layout: dict[str, Any],
    w1_tensor: np.ndarray,
    w2_tensor: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    out = np.zeros((layout["n_samples"], b2.size), dtype=np.float64)
    for item in layout["placements"]:
        i = item["sample_index"]
        batch = item["batch"]
        col = item["col"]
        w1 = np.asarray(w1_tensor[batch, :32, :32], dtype=np.float64).T
        w2 = np.asarray(w2_tensor[batch, :10, :32], dtype=np.float64).T
        z1 = np.asarray(x[i], dtype=np.float64) @ w1 + np.asarray(b1, dtype=np.float64)
        h1 = eval_power_polynomial(z1, coeffs)
        out[i] = h1 @ w2 + np.asarray(b2, dtype=np.float64)
    return out


def mutation_results(*, n_samples: int, degree: int = 3, mutation: str = "correct_weight") -> dict[str, Any]:
    from desilofhe import GLEngine

    t_total = now_seconds()
    model, arrays = load_pca32_model()
    n = min(n_samples, arrays["x_test_32"].shape[0])
    x = arrays["x_test_32"][:n]
    y = arrays["y_test"][:n]
    z_train = model.relu_forward(arrays["x_train_32"])["z1"]
    radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z_train), 0.995)))))
    coeffs = fit_relu_power_polynomial(degree, (-radius, radius)).astype(np.float64)

    engine = GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    had_key = engine.create_hadamard_multiplication_key(sk)

    input_tensor, layout = pack_columns(x, shape)
    used_batches = layout["used_batches"]
    w1_tensor = broadcast_weight_to_batches(model.w1.T, shape, used_batches, 32, 32)
    w2_tensor = broadcast_weight_to_batches(model.w2.T, shape, used_batches, 10, 32)
    b1_pt = engine.encode(broadcast_bias_to_columns(model.b1, shape, layout, 32))
    b2_pt = engine.encode(broadcast_bias_to_columns(model.b2, shape, layout, 10))

    mutated_w1 = np.array(w1_tensor, copy=True)
    mutated_w2 = np.array(w2_tensor, copy=True)
    mutation_record: dict[str, Any] = {"mutation": mutation}
    timing: dict[str, float] = {}
    level_log: list[dict[str, Any]] = []

    if mutation == "zero_W1":
        mutated_w1[...] = 0.0
    elif mutation == "zero_W2":
        mutated_w2[...] = 0.0
    elif mutation == "random_W1":
        rng = np.random.default_rng(12345)
        mutated_w1 = rng.normal(0.0, 0.12, size=shape)
    elif mutation == "shuffled_W1_batches":
        if used_batches > 1:
            mutated_w1[0] = np.array(w1_tensor[0], copy=True)
            mutated_w1[1] = np.array(w1_tensor[1], copy=True)
            mutated_w1[0, :32, :32] += 0.01
            mutated_w1[1, :32, :32] -= 0.01
            mutated_w1[[0, 1]] = mutated_w1[[1, 0]]
            mutation_record["batch_swap"] = [0, 1]
        else:
            perm = np.roll(np.arange(32), 1)
            mutated_w1[0, :32, :32] = mutated_w1[0, perm][:, perm]
            mutation_record["batch_shuffle"] = perm.tolist()
    elif mutation == "wrong_key_or_wrong_weight":
        mutation_record["attempted"] = "wrong_key"
        alt_sk = engine.create_secret_key()
        t_wrong = now_seconds()
        try:
            ct_w1_wrong = engine.encrypt(mutated_w1, alt_sk)
            ct_w2_wrong = engine.encrypt(mutated_w2, alt_sk)
            timing["encryption_weight_s"] = elapsed(t_wrong)
        except Exception as exc:
            mutation_record["fallback"] = "wrong_weight"
            mutation_record["fallback_exception"] = f"{type(exc).__name__}: {exc}"
            rng = np.random.default_rng(54321)
            mutated_w1 = rng.normal(0.0, 0.12, size=shape)
            mutated_w2 = np.array(w2_tensor, copy=True)
            mutated_w2[0] = 0.0
            t = now_seconds()
            ct_w1_wrong = engine.encrypt(mutated_w1, sk)
            ct_w2_wrong = engine.encrypt(mutated_w2, sk)
            timing["encryption_weight_s"] = elapsed(t)
    elif mutation != "correct_weight":
        raise ValueError(f"unknown mutation {mutation}")

    t = now_seconds()
    ct_x = engine.encrypt(input_tensor, sk)
    timing["encryption_input_s"] = elapsed(t)
    level_log.append({"stage": "input", **ct_info(ct_x)})
    t = now_seconds()
    if mutation == "wrong_key_or_wrong_weight":
        ct_w1 = ct_w1_wrong
        ct_w2 = ct_w2_wrong
    else:
        ct_w1 = engine.encrypt(mutated_w1, sk)
        ct_w2 = engine.encrypt(mutated_w2, sk)
        timing["encryption_weight_s"] = elapsed(t)
    level_log.append({"stage": "weight_w1", **ct_info(ct_w1)})
    level_log.append({"stage": "weight_w2", **ct_info(ct_w2)})
    t = now_seconds()
    try:
        ct_h = engine.matrix_multiply(ct_w1, ct_x, mm_key)
        ct_h = engine.add(ct_h, b1_pt)
        timing["linear1_matrix_multiply_s"] = elapsed(t)
        level_log.append({"stage": "linear1", **ct_info(ct_h)})
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
        correct_plain = model.poly_forward(x, coeffs)["logits"]
        mutated_plain = packed_plaintext_logits(x, layout, mutated_w1, mutated_w2, model.b1, model.b2, coeffs)
        against_correct = scalar_metrics(correct_plain, logits, "correct")
        against_mutated = scalar_metrics(mutated_plain, logits, "mutated")
        runtime_total = elapsed(t_total)
        return {
            "ok": True,
            "mutation": mutation,
            "n_samples": n,
            "degree": degree,
            "shape": shape,
            "used_batches": used_batches,
            "semantic_validation_passed_against_correct": bool(against_correct["correct_relative_l2"] < 1e-5 and against_correct["correct_allclose"]),
            "semantic_validation_passed_against_mutated": bool(against_mutated["mutated_relative_l2"] < 1e-5 and against_mutated["mutated_allclose"]),
            "correct_relative_l2": against_correct["correct_relative_l2"],
            "mutated_relative_l2": against_mutated["mutated_relative_l2"],
            "argmax_agreement_against_correct": argmax_agreement(correct_plain, logits),
            "argmax_agreement_against_mutated": argmax_agreement(mutated_plain, logits),
            "ct_W1_type": type(ct_w1).__name__,
            "ct_W2_type": type(ct_w2).__name__,
            "ct_X_type": type(ct_x).__name__,
            "matrix_multiply_operand_kinds": {
                "W1": operand_kind(ct_w1),
                "W2": operand_kind(ct_w2),
                "X": operand_kind(ct_x),
            },
            "weight_encryption_s": timing.get("encryption_weight_s"),
            "input_encryption_s": timing.get("encryption_input_s"),
            "server_only_s": (timing.get("linear1_matrix_multiply_s") or 0.0) + (timing.get("activation_s") or 0.0) + (timing.get("linear2_matrix_multiply_s") or 0.0),
            "total_s": runtime_total,
            "runtime_total_s": runtime_total,
            "runtime_per_sample_s": runtime_total / n,
            "timing_log": timing,
            "level_log": level_log,
            "mutation_record": mutation_record,
        }
    except Exception as exc:
        runtime_total = elapsed(t_total)
        return {
            "ok": False,
            "mutation": mutation,
            "n_samples": n,
            "degree": degree,
            "shape": shape,
            "used_batches": used_batches,
            "semantic_validation_passed_against_correct": False,
            "semantic_validation_passed_against_mutated": False,
            "correct_relative_l2": None,
            "mutated_relative_l2": None,
            "argmax_agreement_against_correct": None,
            "argmax_agreement_against_mutated": None,
            "ct_W1_type": type(ct_w1).__name__ if "ct_w1" in locals() else None,
            "ct_W2_type": type(ct_w2).__name__ if "ct_w2" in locals() else None,
            "ct_X_type": type(ct_x).__name__ if "ct_x" in locals() else None,
            "matrix_multiply_operand_kinds": {
                "W1": operand_kind(ct_w1) if "ct_w1" in locals() else None,
                "W2": operand_kind(ct_w2) if "ct_w2" in locals() else None,
                "X": operand_kind(ct_x) if "ct_x" in locals() else None,
            },
            "weight_encryption_s": timing.get("encryption_weight_s") if "timing" in locals() else None,
            "input_encryption_s": timing.get("encryption_input_s") if "timing" in locals() else None,
            "server_only_s": (timing.get("linear1_matrix_multiply_s") or 0.0) + (timing.get("activation_s") or 0.0) + (timing.get("linear2_matrix_multiply_s") or 0.0) if "timing" in locals() else None,
            "total_s": runtime_total,
            "runtime_total_s": runtime_total,
            "runtime_per_sample_s": runtime_total / n,
            "timing_log": timing if "timing" in locals() else None,
            "level_log": level_log if "level_log" in locals() else None,
            "mutation_record": mutation_record,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "failure_category": classify_failure(exc),
        }
