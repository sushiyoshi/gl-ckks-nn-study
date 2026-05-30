from __future__ import annotations

from typing import Any

import numpy as np

from .gl_block_linear import (
    apply_polynomial_to_blocks,
    bias_block_tensor,
    block_linear_encrypted,
    block_linear_plain_weight_encrypted_input,
    block_linear_plain_gl_layout,
    broadcast_weight_blocks_to_gl,
    pack_sample_columns_for_blocks,
    split_features_to_blocks,
    split_weight_matrix_for_column_vector_eval,
    unpack_output_blocks,
)
from .gl_block_mlp import classify_failure
from .gl_block_schedule import mlp_block_schedule
from .gl_packing import packing_stats
from .gl_shape_selector import mlp_utilization_stats
from .logging_utils import ct_info, elapsed, exception_record, now_seconds
from .metrics import error_metrics
from .polynomial import eval_power_polynomial
from .precision_audit import summarize_array, tensor_error_metrics


def failure_payload(exc: BaseException, **context: Any) -> dict[str, Any]:
    payload = exception_record(exc, **context)
    payload["semantic_validation_passed"] = False
    payload["failure_category"] = classify_failure(exc)
    return payload


def logical_poly_mlp(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    coeffs: np.ndarray,
) -> dict[str, np.ndarray]:
    z1 = np.asarray(x, dtype=np.float64) @ np.asarray(w1, dtype=np.float64) + np.asarray(b1, dtype=np.float64)
    h1 = eval_power_polynomial(z1, coeffs)
    logits = h1 @ np.asarray(w2, dtype=np.float64) + np.asarray(b2, dtype=np.float64)
    return {"z1": z1, "h1": h1, "logits": logits}


def validate_plain_gl_layout(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    coeffs: np.ndarray,
    shape: tuple[int, int, int] = (256, 32, 32),
    block_size: int = 32,
) -> dict[str, Any]:
    n_in, n_hidden = w1.shape
    hidden_in, n_out = w2.shape
    if hidden_in != n_hidden:
        raise ValueError(f"w2 input dim {hidden_in} does not match hidden dim {n_hidden}")

    input_blocks, layout = pack_sample_columns_for_blocks(split_features_to_blocks(x, block_size), shape)
    w1_blocks = split_weight_matrix_for_column_vector_eval(w1, n_in, n_hidden, block_size)
    hidden_blocks, hidden = block_linear_plain_gl_layout(input_blocks, w1_blocks, b1, n_hidden, layout, shape, block_size)
    hidden_poly_blocks = [eval_power_polynomial(block, coeffs) for block in hidden_blocks]
    hidden_poly = unpack_output_blocks(hidden_poly_blocks, layout, n_hidden, block_size)
    w2_blocks = split_weight_matrix_for_column_vector_eval(w2, n_hidden, n_out, block_size)
    output_blocks, logits = block_linear_plain_gl_layout(hidden_poly_blocks, w2_blocks, b2, n_out, layout, shape, block_size)

    logical = logical_poly_mlp(x, w1, b1, w2, b2, coeffs)
    z1_err = error_metrics(logical["z1"], hidden, "plain_gl_z1")
    h1_err = error_metrics(logical["h1"], hidden_poly, "plain_gl_h1")
    logits_err = error_metrics(logical["logits"], logits, "plain_gl_logits")
    return {
        "ok": bool(
            np.allclose(logical["z1"], hidden, rtol=1e-10, atol=1e-10)
            and np.allclose(logical["h1"], hidden_poly, rtol=1e-10, atol=1e-10)
            and np.allclose(logical["logits"], logits, rtol=1e-10, atol=1e-10)
        ),
        "layout": layout,
        "logical": logical,
        "gl_logits": logits,
        "gl_hidden": hidden,
        "gl_hidden_poly": hidden_poly,
        "output_blocks": output_blocks,
        **z1_err,
        **h1_err,
        **logits_err,
    }


def run_encrypted_weight_generic_mlp(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    coeffs: np.ndarray,
    polynomial_radius: float,
    y: np.ndarray | None = None,
    task_type: str = "generic_mlp_poly_relu",
    metadata: dict[str, Any] | None = None,
    max_block_matmuls: int | None = None,
    shape: tuple[int, int, int] | None = None,
    block_size: int = 32,
) -> dict[str, Any]:
    from desilofhe import GLEngine

    t_total = now_seconds()
    x = np.asarray(x, dtype=np.float64)
    w1 = np.asarray(w1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    n_samples, n_in = x.shape
    w1_in, n_hidden = w1.shape
    w2_in, n_out = w2.shape
    if w1_in != n_in or w2_in != n_hidden:
        raise ValueError(f"incompatible dims x={x.shape}, w1={w1.shape}, w2={w2.shape}")

    requested_shape = tuple(int(v) for v in shape) if shape is not None else None
    engine = GLEngine(shape=requested_shape) if requested_shape is not None else GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    if requested_shape is not None and shape != requested_shape:
        raise RuntimeError(f"GLEngine returned shape {shape}, requested {requested_shape}")
    if shape[1] != shape[2]:
        raise ValueError(f"generic MLP requires square GL block rows/cols, got {shape[1:]}")
    if shape[1] < block_size or shape[2] < block_size:
        raise ValueError(f"block_size {block_size} exceeds GL matrix shape {shape[1:]}")
    schedule = mlp_block_schedule(n_in, n_hidden, n_out, n_samples, block_size=block_size, shape=shape)
    if max_block_matmuls is not None and schedule["total_linear_block_matmuls"] > max_block_matmuls:
        raise RuntimeError(
            f"block matmul count {schedule['total_linear_block_matmuls']} exceeds max_block_matmuls={max_block_matmuls}"
        )

    plain_validation = validate_plain_gl_layout(x, w1, b1, w2, b2, coeffs, shape=shape, block_size=block_size)
    if not plain_validation["ok"]:
        raise RuntimeError("plain GL-layout validation failed")
    logits_plain = plain_validation["logical"]["logits"]

    input_blocks, layout = pack_sample_columns_for_blocks(split_features_to_blocks(x, block_size), shape)
    w1_blocks = split_weight_matrix_for_column_vector_eval(w1, n_in, n_hidden, block_size)
    w2_blocks = split_weight_matrix_for_column_vector_eval(w2, n_hidden, n_out, block_size)
    w1_tensors = broadcast_weight_blocks_to_gl(w1_blocks, shape, layout["used_batches"])
    w2_tensors = broadcast_weight_blocks_to_gl(w2_blocks, shape, layout["used_batches"])

    timing: dict[str, float] = {}
    level_log: list[dict[str, Any]] = []
    t = now_seconds()
    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    had_key = engine.create_hadamard_multiplication_key(sk)
    key_generation_s = elapsed(t)

    t = now_seconds()
    ct_inputs = [engine.encrypt(block, sk) for block in input_blocks]
    timing["encryption_input_s"] = elapsed(t)
    for i, ct in enumerate(ct_inputs):
        level_log.append({"stage": f"input_block{i}", **ct_info(ct)})

    t = now_seconds()
    ct_w1 = [[engine.encrypt(block, sk) for block in row] for row in w1_tensors]
    ct_w2 = [[engine.encrypt(block, sk) for block in row] for row in w2_tensors]
    timing["encryption_weight_s"] = elapsed(t)
    for j, row in enumerate(ct_w1):
        for i, ct in enumerate(row):
            level_log.append({"stage": f"weight_w1_out{j}_in{i}", **ct_info(ct)})
    for j, row in enumerate(ct_w2):
        for i, ct in enumerate(row):
            level_log.append({"stage": f"weight_w2_out{j}_in{i}", **ct_info(ct)})

    b1_plain = [
        engine.encode(bias_block_tensor(b1, j, layout, shape, n_hidden, block_size))
        for j in range(schedule["hidden_blocks"])
    ]
    b2_plain = [
        engine.encode(bias_block_tensor(b2, j, layout, shape, n_out, block_size))
        for j in range(schedule["output_blocks"])
    ]

    ct_hidden = block_linear_encrypted(
        engine,
        ct_inputs,
        ct_w1,
        b1_plain,
        mm_key,
        level_log=level_log,
        timing_log=timing,
        stage_prefix="linear1",
        timer=now_seconds,
        elapsed_fn=elapsed,
        ct_info_fn=ct_info,
    )
    ct_activated = apply_polynomial_to_blocks(
        engine,
        ct_hidden,
        coeffs,
        had_key,
        level_log=level_log,
        timing_log=timing,
        timer=now_seconds,
        elapsed_fn=elapsed,
        ct_info_fn=ct_info,
    )
    ct_outputs = block_linear_encrypted(
        engine,
        ct_activated,
        ct_w2,
        b2_plain,
        mm_key,
        level_log=level_log,
        timing_log=timing,
        stage_prefix="linear2",
        timer=now_seconds,
        elapsed_fn=elapsed,
        ct_info_fn=ct_info,
    )

    t = now_seconds()
    dec_blocks = [np.asarray(engine.decrypt(ct, sk), dtype=np.float64) for ct in ct_outputs]
    timing["decryption_s"] = elapsed(t)
    logits = unpack_output_blocks(dec_blocks, layout, n_out, block_size)

    err = error_metrics(logits_plain, logits, "logits")
    logits_allclose = bool(np.allclose(logits, logits_plain, rtol=1e-5, atol=1e-5))
    runtime_total = elapsed(t_total)
    server_only_s = sum(
        value
        for key, value in timing.items()
        if key.endswith("_matrix_multiply_s") or key.startswith("activation_block")
    )
    total_minus_keygen_s = runtime_total - key_generation_s
    accuracy_value = None
    argmax_agreement = None
    if y is not None:
        from .metrics import accuracy as accuracy_fn
        from .metrics import argmax_agreement as agreement_fn

        accuracy_value = accuracy_fn(logits, np.asarray(y))
        argmax_agreement = agreement_fn(logits_plain, logits)

    return {
        "ok": True,
        "semantic_validation_passed": bool(err["logits_relative_l2"] < 1e-5 and logits_allclose),
        "failure_category": None,
        "task_type": task_type,
        "weight_privacy": "encrypted_weight",
        "activation": "degree3_poly",
        "polynomial_degree": int(len(coeffs) - 1),
        "polynomial_radius": float(polynomial_radius),
        "coefficients": coeffs.tolist(),
        "dims": [n_in, n_hidden, n_out],
        "logical_dims": [n_in, n_hidden, n_out],
        "n_samples": n_samples,
        "block_size": block_size,
        "shape": list(shape),
        "selected_shape": list(shape),
        "sample_capacity": shape[0] * shape[2],
        "schedule": schedule,
        "ciphertext_count": schedule["ciphertext_counts"]["input"]
        + schedule["ciphertext_counts"]["W1"]
        + schedule["ciphertext_counts"]["W2"],
        "input_ciphertext_count": schedule["ciphertext_counts"]["input"],
        "weight_ciphertext_count": schedule["ciphertext_counts"]["W1"] + schedule["ciphertext_counts"]["W2"],
        "block_matmul_count": schedule["total_linear_block_matmuls"],
        "used_batches": layout["used_batches"],
        "used_columns_last_batch": layout["used_columns_last_batch"],
        **packing_stats(n_samples, n_in, shape),
        **mlp_utilization_stats(n_samples, n_in, n_hidden, n_out, shape, block_size),
        "plain_gl_layout_validation": {
            key: value
            for key, value in plain_validation.items()
            if key not in {"logical", "gl_logits", "gl_hidden", "gl_hidden_poly", "output_blocks", "layout"}
        },
        "metadata": metadata or {},
        **err,
        "relative_l2": err["logits_relative_l2"],
        "linf": err["logits_linf"],
        "mae": err["logits_mae"],
        "allclose": logits_allclose,
        "logits_allclose": logits_allclose,
        "accuracy": accuracy_value,
        "argmax_agreement": argmax_agreement,
        "level_log": level_log,
        "timing_log": timing,
        "key_generation_s": key_generation_s,
        "server_only_s": server_only_s,
        "total_s": runtime_total,
        "runtime_total_s": runtime_total,
        "runtime_per_sample_s": runtime_total / n_samples,
        "total_minus_keygen_s": total_minus_keygen_s,
        "server_per_sample_s": server_only_s / n_samples,
        "no_keygen_per_sample_s": total_minus_keygen_s / n_samples,
    }


def run_plain_weight_generic_mlp(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    coeffs: np.ndarray,
    polynomial_radius: float,
    y: np.ndarray | None = None,
    task_type: str = "generic_mlp_poly_relu",
    metadata: dict[str, Any] | None = None,
    max_block_matmuls: int | None = None,
    shape: tuple[int, int, int] | None = None,
    block_size: int = 32,
) -> dict[str, Any]:
    from desilofhe import GLEngine

    t_total = now_seconds()
    x = np.asarray(x, dtype=np.float64)
    w1 = np.asarray(w1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    n_samples, n_in = x.shape
    w1_in, n_hidden = w1.shape
    w2_in, n_out = w2.shape
    if w1_in != n_in or w2_in != n_hidden:
        raise ValueError(f"incompatible dims x={x.shape}, w1={w1.shape}, w2={w2.shape}")

    requested_shape = tuple(int(v) for v in shape) if shape is not None else None
    engine = GLEngine(shape=requested_shape) if requested_shape is not None else GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    if requested_shape is not None and shape != requested_shape:
        raise RuntimeError(f"GLEngine returned shape {shape}, requested {requested_shape}")
    if shape[1] != shape[2]:
        raise ValueError(f"generic MLP requires square GL block rows/cols, got {shape[1:]}")
    if shape[1] < block_size or shape[2] < block_size:
        raise ValueError(f"block_size {block_size} exceeds GL matrix shape {shape[1:]}")
    schedule = mlp_block_schedule(n_in, n_hidden, n_out, n_samples, block_size=block_size, shape=shape)
    if max_block_matmuls is not None and schedule["total_linear_block_matmuls"] > max_block_matmuls:
        raise RuntimeError(
            f"block matmul count {schedule['total_linear_block_matmuls']} exceeds max_block_matmuls={max_block_matmuls}"
        )

    plain_validation = validate_plain_gl_layout(x, w1, b1, w2, b2, coeffs, shape=shape, block_size=block_size)
    if not plain_validation["ok"]:
        raise RuntimeError("plain GL-layout validation failed")
    logits_plain = plain_validation["logical"]["logits"]

    input_blocks, layout = pack_sample_columns_for_blocks(split_features_to_blocks(x, block_size), shape)
    w1_blocks = split_weight_matrix_for_column_vector_eval(w1, n_in, n_hidden, block_size)
    w2_blocks = split_weight_matrix_for_column_vector_eval(w2, n_hidden, n_out, block_size)
    w1_tensors = broadcast_weight_blocks_to_gl(w1_blocks, shape, layout["used_batches"])
    w2_tensors = broadcast_weight_blocks_to_gl(w2_blocks, shape, layout["used_batches"])

    timing: dict[str, float] = {}
    level_log: list[dict[str, Any]] = []
    t = now_seconds()
    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    had_key = engine.create_hadamard_multiplication_key(sk)
    key_generation_s = elapsed(t)

    t = now_seconds()
    ct_inputs = [engine.encrypt(block, sk) for block in input_blocks]
    timing["encryption_input_s"] = elapsed(t)
    for i, ct in enumerate(ct_inputs):
        level_log.append({"stage": f"input_block{i}", **ct_info(ct)})

    t = now_seconds()
    pt_w1 = [[engine.encode(block) for block in row] for row in w1_tensors]
    pt_w2 = [[engine.encode(block) for block in row] for row in w2_tensors]
    timing["plaintext_weight_encode_s"] = elapsed(t)

    b1_plain = [
        engine.encode(bias_block_tensor(b1, j, layout, shape, n_hidden, block_size))
        for j in range(schedule["hidden_blocks"])
    ]
    b2_plain = [
        engine.encode(bias_block_tensor(b2, j, layout, shape, n_out, block_size))
        for j in range(schedule["output_blocks"])
    ]

    ct_hidden = block_linear_plain_weight_encrypted_input(
        engine,
        ct_inputs,
        pt_w1,
        b1_plain,
        mm_key,
        level_log=level_log,
        timing_log=timing,
        stage_prefix="linear1",
        timer=now_seconds,
        elapsed_fn=elapsed,
        ct_info_fn=ct_info,
    )
    ct_activated = apply_polynomial_to_blocks(
        engine,
        ct_hidden,
        coeffs,
        had_key,
        level_log=level_log,
        timing_log=timing,
        timer=now_seconds,
        elapsed_fn=elapsed,
        ct_info_fn=ct_info,
    )
    ct_outputs = block_linear_plain_weight_encrypted_input(
        engine,
        ct_activated,
        pt_w2,
        b2_plain,
        mm_key,
        level_log=level_log,
        timing_log=timing,
        stage_prefix="linear2",
        timer=now_seconds,
        elapsed_fn=elapsed,
        ct_info_fn=ct_info,
    )

    t = now_seconds()
    dec_blocks = [np.asarray(engine.decrypt(ct, sk), dtype=np.float64) for ct in ct_outputs]
    timing["decryption_s"] = elapsed(t)
    logits = unpack_output_blocks(dec_blocks, layout, n_out, block_size)

    err = error_metrics(logits_plain, logits, "logits")
    logits_allclose = bool(np.allclose(logits, logits_plain, rtol=1e-5, atol=1e-5))
    runtime_total = elapsed(t_total)
    server_only_s = sum(
        value
        for key, value in timing.items()
        if key.endswith("_matrix_multiply_s") or key.startswith("activation_block")
    )
    total_minus_keygen_s = runtime_total - key_generation_s
    accuracy_value = None
    argmax_agreement = None
    if y is not None:
        from .metrics import accuracy as accuracy_fn
        from .metrics import argmax_agreement as agreement_fn

        accuracy_value = accuracy_fn(logits, np.asarray(y))
        argmax_agreement = agreement_fn(logits_plain, logits)

    return {
        "ok": True,
        "semantic_validation_passed": bool(err["logits_relative_l2"] < 1e-5 and logits_allclose),
        "failure_category": None,
        "task_type": task_type,
        "weight_privacy": "plaintext_weight",
        "activation": "degree3_poly",
        "polynomial_degree": int(len(coeffs) - 1),
        "polynomial_radius": float(polynomial_radius),
        "coefficients": coeffs.tolist(),
        "dims": [n_in, n_hidden, n_out],
        "logical_dims": [n_in, n_hidden, n_out],
        "n_samples": n_samples,
        "block_size": block_size,
        "shape": list(shape),
        "selected_shape": list(shape),
        "sample_capacity": shape[0] * shape[2],
        "schedule": schedule,
        "ciphertext_count": schedule["ciphertext_counts"]["input"],
        "input_ciphertext_count": schedule["ciphertext_counts"]["input"],
        "weight_ciphertext_count": 0,
        "plaintext_weight_count": schedule["ciphertext_counts"]["W1"] + schedule["ciphertext_counts"]["W2"],
        "block_matmul_count": schedule["total_linear_block_matmuls"],
        "used_batches": layout["used_batches"],
        "used_columns_last_batch": layout["used_columns_last_batch"],
        **packing_stats(n_samples, n_in, shape),
        **mlp_utilization_stats(n_samples, n_in, n_hidden, n_out, shape, block_size),
        "plain_gl_layout_validation": {
            key: value
            for key, value in plain_validation.items()
            if key not in {"logical", "gl_logits", "gl_hidden", "gl_hidden_poly", "output_blocks", "layout"}
        },
        "metadata": metadata or {},
        **err,
        "relative_l2": err["logits_relative_l2"],
        "linf": err["logits_linf"],
        "mae": err["logits_mae"],
        "allclose": logits_allclose,
        "logits_allclose": logits_allclose,
        "accuracy": accuracy_value,
        "argmax_agreement": argmax_agreement,
        "level_log": level_log,
        "timing_log": timing,
        "key_generation_s": key_generation_s,
        "server_only_s": server_only_s,
        "total_s": runtime_total,
        "runtime_total_s": runtime_total,
        "runtime_per_sample_s": runtime_total / n_samples,
        "total_minus_keygen_s": total_minus_keygen_s,
        "server_per_sample_s": server_only_s / n_samples,
        "no_keygen_per_sample_s": total_minus_keygen_s / n_samples,
    }


def _with_level(row: dict[str, Any], source: Any) -> dict[str, Any]:
    if isinstance(source, (list, tuple)):
        levels = [getattr(item, "level", None) for item in source]
        present = [level for level in levels if level is not None]
        row["levels"] = levels
        if present:
            row["level"] = present[0] if len(set(present)) == 1 else min(present)
    else:
        row["level"] = getattr(source, "level", None)
    return row


def _has_direct_noise_budget(cts: list[Any]) -> bool:
    names = ("noise_budget", "noiseBudget", "get_noise_budget", "GetNoiseBudget")
    return any(any(hasattr(ct, name) for name in names) for ct in cts)


def run_precision_audit_generic_mlp(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    coeffs: np.ndarray,
    polynomial_radius: float,
    task_type: str = "generic_mlp_poly_relu",
    metadata: dict[str, Any] | None = None,
    max_block_matmuls: int | None = None,
) -> dict[str, Any]:
    from desilofhe import GLEngine

    x = np.asarray(x, dtype=np.float64)
    w1 = np.asarray(w1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    n_samples, n_in = x.shape
    w1_in, n_hidden = w1.shape
    w2_in, n_out = w2.shape
    if w1_in != n_in or w2_in != n_hidden:
        raise ValueError(f"incompatible dims x={x.shape}, w1={w1.shape}, w2={w2.shape}")

    engine = GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    schedule = mlp_block_schedule(n_in, n_hidden, n_out, n_samples, shape=shape)
    if max_block_matmuls is not None and schedule["total_linear_block_matmuls"] > max_block_matmuls:
        raise RuntimeError(
            f"block matmul count {schedule['total_linear_block_matmuls']} exceeds max_block_matmuls={max_block_matmuls}"
        )

    logical = logical_poly_mlp(x, w1, b1, w2, b2, coeffs)
    input_blocks, layout = pack_sample_columns_for_blocks(split_features_to_blocks(x), shape)
    w1_blocks = split_weight_matrix_for_column_vector_eval(w1, n_in, n_hidden)
    hidden_blocks, hidden = block_linear_plain_gl_layout(input_blocks, w1_blocks, b1, n_hidden, layout, shape)
    hidden_poly_blocks = [eval_power_polynomial(block, coeffs) for block in hidden_blocks]
    hidden_poly = unpack_output_blocks(hidden_poly_blocks, layout, n_hidden)
    w2_blocks = split_weight_matrix_for_column_vector_eval(w2, n_hidden, n_out)
    output_blocks, gl_logits = block_linear_plain_gl_layout(hidden_poly_blocks, w2_blocks, b2, n_out, layout, shape)
    logical_z1_blocks, _ = pack_sample_columns_for_blocks(split_features_to_blocks(logical["z1"]), shape)
    logical_h1_blocks, _ = pack_sample_columns_for_blocks(split_features_to_blocks(logical["h1"]), shape)
    logical_logits_blocks, _ = pack_sample_columns_for_blocks(split_features_to_blocks(logical["logits"]), shape)

    w1_tensors = broadcast_weight_blocks_to_gl(w1_blocks, shape, layout["used_batches"])
    w2_tensors = broadcast_weight_blocks_to_gl(w2_blocks, shape, layout["used_batches"])
    level_log: list[dict[str, Any]] = []

    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    had_key = engine.create_hadamard_multiplication_key(sk)

    ct_inputs = [engine.encrypt(block, sk) for block in input_blocks]
    for i, ct in enumerate(ct_inputs):
        level_log.append({"stage": f"input_block{i}", **ct_info(ct)})
    ct_w1 = [[engine.encrypt(block, sk) for block in row] for row in w1_tensors]
    ct_w2 = [[engine.encrypt(block, sk) for block in row] for row in w2_tensors]
    for j, row in enumerate(ct_w1):
        for i, ct in enumerate(row):
            level_log.append({"stage": f"weight_w1_out{j}_in{i}", **ct_info(ct)})
    for j, row in enumerate(ct_w2):
        for i, ct in enumerate(row):
            level_log.append({"stage": f"weight_w2_out{j}_in{i}", **ct_info(ct)})

    b1_plain = [
        engine.encode(bias_block_tensor(b1, j, layout, shape, n_hidden))
        for j in range(schedule["hidden_blocks"])
    ]
    b2_plain = [
        engine.encode(bias_block_tensor(b2, j, layout, shape, n_out))
        for j in range(schedule["output_blocks"])
    ]
    ct_hidden = block_linear_encrypted(
        engine,
        ct_inputs,
        ct_w1,
        b1_plain,
        mm_key,
        level_log=level_log,
        stage_prefix="linear1",
        ct_info_fn=ct_info,
    )
    ct_activated = apply_polynomial_to_blocks(
        engine,
        ct_hidden,
        coeffs,
        had_key,
        level_log=level_log,
        ct_info_fn=ct_info,
    )
    ct_outputs = block_linear_encrypted(
        engine,
        ct_activated,
        ct_w2,
        b2_plain,
        mm_key,
        level_log=level_log,
        stage_prefix="linear2",
        ct_info_fn=ct_info,
    )

    dec_inputs = [np.asarray(engine.decrypt(ct, sk), dtype=np.float64) for ct in ct_inputs]
    dec_hidden_blocks = [np.asarray(engine.decrypt(ct, sk), dtype=np.float64) for ct in ct_hidden]
    dec_activated_blocks = [np.asarray(engine.decrypt(ct, sk), dtype=np.float64) for ct in ct_activated]
    dec_output_blocks = [np.asarray(engine.decrypt(ct, sk), dtype=np.float64) for ct in ct_outputs]
    dec_hidden = unpack_output_blocks(dec_hidden_blocks, layout, n_hidden)
    dec_activated = unpack_output_blocks(dec_activated_blocks, layout, n_hidden)
    logits = unpack_output_blocks(dec_output_blocks, layout, n_out)

    stages: list[dict[str, Any]] = []
    stages.append({**summarize_array("logical_plain.packed_input_blocks", np.asarray(input_blocks)), "comparison": "summary"})
    stages.append({
        **tensor_error_metrics("packed_input_blocks", np.asarray(input_blocks), np.asarray(dec_inputs)),
        "comparison": "gl_layout_plain_vs_encrypted_decrypted",
    })
    _with_level(stages[-1], ct_inputs)
    for stage, plain, actual in [
        ("linear1_block_outputs", np.asarray(logical_z1_blocks), np.asarray(hidden_blocks)),
        ("linear1_accumulated_output", logical["z1"], hidden),
        ("activation_output", logical["h1"], hidden_poly),
        ("linear2_final_logits", np.asarray(logical_logits_blocks), np.asarray(output_blocks)),
        ("unpacked_logits", logical["logits"], gl_logits),
    ]:
        stages.append({**tensor_error_metrics(stage, plain, actual), "comparison": "logical_plain_vs_gl_layout_plain"})
    for stage, plain, actual, cts in [
        ("encrypted_decrypted_linear1_block_outputs", np.asarray(hidden_blocks), np.asarray(dec_hidden_blocks), ct_hidden),
        ("linear1_accumulated_output", hidden, dec_hidden, ct_hidden),
        ("activation_output", hidden_poly, dec_activated, ct_activated),
        ("linear2_final_logits", np.asarray(output_blocks), np.asarray(dec_output_blocks), ct_outputs),
        ("unpacked_logits", gl_logits, logits, ct_outputs),
    ]:
        row = {**tensor_error_metrics(stage, plain, actual), "comparison": "gl_layout_plain_vs_encrypted_decrypted"}
        stages.append(_with_level(row, cts))

    ciphertexts = ct_inputs + [ct for row in ct_w1 for ct in row] + [ct for row in ct_w2 for ct in row] + ct_hidden + ct_activated + ct_outputs
    noise_note = None if _has_direct_noise_budget(ciphertexts) else "direct noise budget unavailable"
    return {
        "ok": True,
        "audit_precision": True,
        "audit_run_kind": "precision_noise_audit",
        "performance_run": False,
        "ciphertext_payload_saved": False,
        "task_type": task_type,
        "weight_privacy": "encrypted_weight",
        "activation": "degree3_poly",
        "polynomial_degree": int(len(coeffs) - 1),
        "polynomial_radius": float(polynomial_radius),
        "coefficients": coeffs.tolist(),
        "dims": [n_in, n_hidden, n_out],
        "n_samples": n_samples,
        "block_size": 32,
        "shape": list(shape),
        "schedule": schedule,
        "metadata": {
            **(metadata or {}),
            "noise_budget": noise_note,
            "noise_budget_proxy": "level_log and decrypted error metrics",
        },
        "level_log": level_log,
        "stages": stages,
    }
