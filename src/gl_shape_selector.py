from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Iterable

from .gl_block_schedule import mlp_block_schedule


DEFAULT_ALLOWED_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SUPPORTED_GL_SHAPES = [
    (256, 16, 16),
    (256, 32, 32),
    (256, 64, 64),
    (16, 256, 256),
    (16, 512, 512),
    (4, 1024, 1024),
    (4, 2048, 2048),
]
BLOCK_COST_FACTORS = {
    16: 1.0,
    32: 1.0,
    64: 2.0,
    256: 16.0,
    512: 64.0,
    1024: 256.0,
    2048: 1024.0,
}


def choose_batch_count(
    n_samples: int,
    columns: int = 32,
    allowed_batches: Iterable[int] | None = None,
) -> int:
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    if columns <= 0:
        raise ValueError(f"columns must be positive, got {columns}")
    allowed = sorted(int(v) for v in (allowed_batches or DEFAULT_ALLOWED_BATCHES))
    if not allowed:
        raise ValueError("allowed_batches must not be empty")
    required = math.ceil(n_samples / columns) if n_samples else 1
    for batch_count in allowed:
        if batch_count >= required:
            return batch_count
    raise ValueError(
        f"n_samples={n_samples} needs at least {required} batches with columns={columns}; "
        f"largest allowed batch count is {allowed[-1]}"
    )


def choose_shape(
    n_samples: int,
    n_in: int,
    n_hidden: int,
    n_out: int,
    block_size: int | None = 32,
    allowed_batches: Iterable[int] | None = None,
    columns: int | None = None,
    policy: str = "legacy_auto_batches",
    explicit_shape: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    if policy != "legacy_auto_batches":
        return choose_supported_shape(
            n_samples,
            n_in,
            n_hidden,
            n_out,
            policy=policy,
            explicit_shape=explicit_shape,
        )
    if block_size is None:
        block_size = 32
    cols = int(columns if columns is not None else block_size)
    batch_count = choose_batch_count(n_samples, columns=cols, allowed_batches=allowed_batches)
    shape = (batch_count, int(block_size), cols)
    schedule = estimate_schedule_for_shape(n_samples, n_in, n_hidden, n_out, shape=shape, block_size=block_size)
    return {
        "shape": list(shape),
        "batch_count": batch_count,
        "block_size": int(block_size),
        "columns": cols,
        "schedule": schedule,
        "memory_estimate": estimate_memory_for_shape(n_samples, n_in, n_hidden, n_out, shape=shape, block_size=block_size),
        "selection_reason": "minimum allowed batch_count satisfying n_samples <= batch_count * columns",
    }


def normalize_shape(shape: Iterable[int]) -> tuple[int, int, int]:
    values = tuple(int(v) for v in shape)
    if len(values) != 3:
        raise ValueError(f"shape must have three dimensions, got {values}")
    if values[1] != values[2]:
        raise ValueError(f"GLEngine MLP selector requires square block rows/cols, got {values}")
    return values


def shape_candidate(
    n_samples: int,
    n_in: int,
    n_hidden: int,
    n_out: int,
    shape: tuple[int, int, int],
    measured_costs: dict[tuple[int, int, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    shape = normalize_shape(shape)
    batch_count, block_rows, block_cols = shape
    block_size = block_rows
    schedule = estimate_schedule_for_shape(n_samples, n_in, n_hidden, n_out, shape=shape, block_size=block_size)
    utilization = mlp_utilization_stats(n_samples, n_in, n_hidden, n_out, shape, block_size)
    memory = estimate_memory_for_shape(n_samples, n_in, n_hidden, n_out, shape=shape, block_size=block_size)
    feature_product = (
        utilization["feature_padding_utilization_input"]
        * utilization["feature_padding_utilization_hidden"]
        * utilization["feature_padding_utilization_output"]
    )
    measured = (measured_costs or {}).get(shape)
    cost_factor = BLOCK_COST_FACTORS.get(block_size, float(block_size * block_size) / (32.0 * 32.0))
    cost_source = "fallback"
    if measured is not None:
        measured_cost = measured.get("matrix_multiply_s")
        if measured_cost is not None and float(measured_cost) > 0:
            cost_factor = float(measured_cost)
            cost_source = "measured_matrix_multiply_s"
    score = (
        schedule["total_linear_block_matmuls"]
        * cost_factor
        / max(utilization["sample_packing_utilization"], 1e-12)
        / max(feature_product, 1e-12)
    )
    return {
        "shape": list(shape),
        "batch_count": batch_count,
        "block_size": block_size,
        "block_rows": block_rows,
        "block_cols": block_cols,
        "sample_capacity": batch_count * block_cols,
        "schedule": schedule,
        "memory_estimate": memory,
        "utilization": utilization,
        "total_block_matmuls": schedule["total_linear_block_matmuls"],
        "activation_blocks": schedule["activation_blocks"],
        "estimated_single_matmul_cost_factor": cost_factor,
        "cost_factor_source": cost_source,
        "measured_probe": measured,
        "feature_padding_utilization_product": feature_product,
        "balanced_score": score,
    }


def load_measured_shape_costs(path: str | Path | None) -> dict[tuple[int, int, int], dict[str, Any]]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text())
    costs: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in payload.get("results", []):
        shape_values = row.get("engine_shape") or row.get("requested_shape")
        if not shape_values:
            continue
        shape = normalize_shape(shape_values)
        if row.get("ok") or row.get("matrix_multiply_ok"):
            costs[shape] = {
                "ok": row.get("ok"),
                "matrix_multiply_s": row.get("matrix_multiply_s"),
                "evaluate_polynomial_s": row.get("evaluate_polynomial_s"),
                "keygen_s": row.get("keygen_s"),
                "source": str(p),
            }
    return costs


def supported_shape_candidates(
    n_samples: int,
    n_in: int,
    n_hidden: int,
    n_out: int,
    supported_shapes: Iterable[tuple[int, int, int]] | None = None,
    measured_costs: dict[tuple[int, int, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidates = []
    for shape in supported_shapes or SUPPORTED_GL_SHAPES:
        shape = normalize_shape(shape)
        if n_samples > shape[0] * shape[2]:
            continue
        candidate = shape_candidate(n_samples, n_in, n_hidden, n_out, shape, measured_costs=measured_costs)
        candidates.append(candidate)
    return candidates


def choose_supported_shape(
    n_samples: int,
    n_in: int,
    n_hidden: int,
    n_out: int,
    policy: str = "balanced",
    explicit_shape: tuple[int, int, int] | None = None,
    supported_shapes: Iterable[tuple[int, int, int]] | None = None,
    probe_results_path: str | Path | None = None,
) -> dict[str, Any]:
    measured_costs = load_measured_shape_costs(probe_results_path)
    if policy == "fixed_32":
        selected = shape_candidate(n_samples, n_in, n_hidden, n_out, (256, 32, 32), measured_costs=measured_costs)
    elif policy == "exact":
        if explicit_shape is None:
            raise ValueError("policy exact requires explicit_shape")
        shape = normalize_shape(explicit_shape)
        if shape not in tuple(supported_shapes or SUPPORTED_GL_SHAPES):
            raise ValueError(f"shape {shape} is not in supported GLEngine shapes")
        selected = shape_candidate(n_samples, n_in, n_hidden, n_out, shape, measured_costs=measured_costs)
        if n_samples > selected["sample_capacity"]:
            raise ValueError(f"n_samples={n_samples} exceeds sample capacity {selected['sample_capacity']} for shape {shape}")
    else:
        candidates = supported_shape_candidates(n_samples, n_in, n_hidden, n_out, supported_shapes, measured_costs=measured_costs)
        if not candidates:
            raise ValueError(f"no supported GLEngine shape can pack n_samples={n_samples}")
        if policy == "min_block_matmuls":
            selected = min(
                candidates,
                key=lambda item: (
                    item["total_block_matmuls"],
                    -item["utilization"]["sample_packing_utilization"],
                    item["memory_estimate"]["logical_tensor_mib_f64_estimate"],
                ),
            )
        elif policy == "balanced":
            selected = min(candidates, key=lambda item: item["balanced_score"])
        else:
            raise ValueError(f"unknown shape policy {policy!r}")
    selected = dict(selected)
    selected["policy"] = policy
    selected["single_pack_only"] = True
    selected["candidate_filter"] = "n_samples <= sample_capacity"
    selected["measured_costs_path"] = str(probe_results_path) if probe_results_path is not None else None
    selected["measured_costs_loaded"] = bool(measured_costs)
    selected["selection_reason"] = {
        "fixed_32": "legacy fixed shape",
        "exact": "explicit supported shape",
        "min_block_matmuls": "minimum total linear block matmul count among supported shapes that fit n_samples",
        "balanced": "minimum simple score using block matmuls, cost factor, sample utilization, and padding utilization",
    }.get(policy, "selected shape")
    return selected


def estimate_schedule_for_shape(
    n_samples: int,
    n_in: int,
    n_hidden: int,
    n_out: int,
    shape: tuple[int, int, int],
    block_size: int = 32,
) -> dict[str, Any]:
    return mlp_block_schedule(n_in, n_hidden, n_out, n_samples, block_size=block_size, shape=shape)


def estimate_memory_for_shape(
    n_samples: int,
    n_in: int,
    n_hidden: int,
    n_out: int,
    shape: tuple[int, int, int],
    block_size: int = 32,
) -> dict[str, Any]:
    schedule = estimate_schedule_for_shape(n_samples, n_in, n_hidden, n_out, shape=shape, block_size=block_size)
    entries_per_tensor = int(shape[0] * shape[1] * shape[2])
    bytes_per_tensor_f64 = entries_per_tensor * 8
    peak_ciphertext_count_estimate = (
        schedule["ciphertext_counts"]["input"]
        + schedule["ciphertext_counts"]["W1"]
        + schedule["ciphertext_counts"]["hidden"]
        + schedule["ciphertext_counts"]["W2"]
        + schedule["ciphertext_counts"]["output"]
    )
    return {
        "entries_per_tensor": entries_per_tensor,
        "bytes_per_tensor_f64": bytes_per_tensor_f64,
        "mib_per_tensor_f64": bytes_per_tensor_f64 / (1024 * 1024),
        "peak_ciphertext_count_estimate": peak_ciphertext_count_estimate,
        "logical_tensor_entries_estimate": entries_per_tensor * peak_ciphertext_count_estimate,
        "logical_tensor_mib_f64_estimate": entries_per_tensor * peak_ciphertext_count_estimate * 8 / (1024 * 1024),
    }


def mlp_utilization_stats(
    n_samples: int,
    n_in: int,
    n_hidden: int,
    n_out: int,
    shape: tuple[int, int, int],
    block_size: int = 32,
) -> dict[str, Any]:
    batch_count, rows, columns = (int(v) for v in shape)
    if rows < block_size or columns <= 0 or batch_count <= 0:
        raise ValueError(f"invalid shape/block_size combination: shape={shape}, block_size={block_size}")
    input_blocks = math.ceil(n_in / block_size) if n_in else 1
    hidden_blocks = math.ceil(n_hidden / block_size) if n_hidden else 1
    output_blocks = math.ceil(n_out / block_size) if n_out else 1
    sample_capacity = batch_count * columns
    sample_packs = math.ceil(n_samples / sample_capacity) if n_samples else 1

    def bounded(feature_dim: int, blocks: int) -> float:
        denominator = sample_packs * blocks * batch_count * block_size * columns
        return (n_samples * feature_dim) / denominator if denominator else 0.0

    stats = {
        "sample_packs": sample_packs,
        "sample_capacity_per_pack": sample_capacity,
        "sample_packing_utilization": n_samples / (sample_packs * sample_capacity) if sample_capacity else 0.0,
        "feature_padding_utilization_input": n_in / (input_blocks * block_size),
        "feature_padding_utilization_hidden": n_hidden / (hidden_blocks * block_size),
        "feature_padding_utilization_output": n_out / (output_blocks * block_size),
        "input_entry_utilization_bounded": bounded(n_in, input_blocks),
        "hidden_entry_utilization_bounded": bounded(n_hidden, hidden_blocks),
        "output_entry_utilization_bounded": bounded(n_out, output_blocks),
        "legacy_matrix_entry_utilization_input_single_tensor_basis": (n_samples * n_in) / (batch_count * rows * columns),
    }
    bounded_metric_keys = {
        "sample_packing_utilization",
        "feature_padding_utilization_input",
        "feature_padding_utilization_hidden",
        "feature_padding_utilization_output",
        "input_entry_utilization_bounded",
        "hidden_entry_utilization_bounded",
        "output_entry_utilization_bounded",
    }
    for key in bounded_metric_keys:
        value = stats[key]
        if not 0.0 <= float(value) <= 1.0:
            raise AssertionError(f"{key}={value} is outside [0, 1]")
    return stats
