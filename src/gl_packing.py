from __future__ import annotations

import math
from typing import Any

import numpy as np


def pack_columns(samples: np.ndarray, shape: tuple[int, int, int], max_cols: int = 32) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"samples must be 2D, got {x.shape}")
    n_samples, feature_dim = x.shape
    batches, rows, cols = shape
    if feature_dim > rows:
        raise ValueError(f"feature_dim {feature_dim} exceeds GL rows {rows}")
    if max_cols > cols:
        raise ValueError(f"max_cols {max_cols} exceeds GL cols {cols}")
    used_batches = math.ceil(n_samples / max_cols) if n_samples else 0
    if used_batches > batches:
        raise ValueError(f"{n_samples} samples need {used_batches} batches, exceeds GL batch axis {batches}")
    tensor = np.zeros(shape, dtype=np.float64)
    placements = []
    for i, sample in enumerate(x):
        batch = i // max_cols
        col = i % max_cols
        tensor[batch, :feature_dim, col] = sample
        placements.append({"sample_index": i, "batch": batch, "col": col})
    layout = {
        "n_samples": n_samples,
        "feature_dim": feature_dim,
        "max_cols": max_cols,
        "used_batches": used_batches,
        "used_columns_last_batch": 0 if n_samples == 0 else ((n_samples - 1) % max_cols) + 1,
        "samples_per_ciphertext_capacity": batches * max_cols,
        "placements": placements,
    }
    return tensor, layout


def unpack_logits(dec: np.ndarray, layout: dict[str, Any], n_logits: int) -> np.ndarray:
    out = np.zeros((layout["n_samples"], n_logits), dtype=np.float64)
    for item in layout["placements"]:
        i = item["sample_index"]
        out[i] = np.asarray(dec, dtype=np.float64)[item["batch"], :n_logits, item["col"]]
    return out


def broadcast_weight_to_batches(
    w_left: np.ndarray,
    shape: tuple[int, int, int],
    used_batches: int,
    out_rows: int | None = None,
    in_rows: int | None = None,
) -> np.ndarray:
    w = np.asarray(w_left, dtype=np.float64)
    out = w.shape[0] if out_rows is None else out_rows
    inn = w.shape[1] if in_rows is None else in_rows
    if out > shape[1] or inn > shape[2]:
        raise ValueError(f"weight logical shape {(out, inn)} exceeds GL matrix {shape[1:]}")
    tensor = np.zeros(shape, dtype=np.float64)
    for batch in range(used_batches):
        tensor[batch, :out, :inn] = w[:out, :inn]
    return tensor


def broadcast_bias_to_columns(bias: np.ndarray, shape: tuple[int, int, int], layout: dict[str, Any], rows: int | None = None) -> np.ndarray:
    b = np.asarray(bias, dtype=np.float64).ravel()
    n_rows = b.size if rows is None else rows
    tensor = np.zeros(shape, dtype=np.float64)
    for item in layout["placements"]:
        tensor[item["batch"], :n_rows, item["col"]] = b[:n_rows]
    return tensor


def split64_to_two_packed_tensors(samples64: np.ndarray, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x = np.asarray(samples64, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 64:
        raise ValueError(f"expected samples shape (N, 64), got {x.shape}")
    x0, layout = pack_columns(x[:, :32], shape)
    x1, layout1 = pack_columns(x[:, 32:64], shape)
    if layout["placements"] != layout1["placements"]:
        raise RuntimeError("split packing layouts diverged")
    return x0, x1, layout


def packing_stats(n_samples: int, feature_dim: int, shape: tuple[int, int, int]) -> dict[str, Any]:
    total_entries = int(np.prod(shape))
    sample_capacity = shape[0] * shape[2]
    used_entries = n_samples * feature_dim
    return {
        "n_samples": n_samples,
        "feature_dim": feature_dim,
        "used_batches": math.ceil(n_samples / shape[2]) if n_samples else 0,
        "used_columns_last_batch": 0 if n_samples == 0 else ((n_samples - 1) % shape[2]) + 1,
        "samples_per_ciphertext_capacity": sample_capacity,
        "sample_packing_utilization": n_samples / sample_capacity,
        "matrix_entry_utilization_input": used_entries / total_entries,
        "used_input_entries": used_entries,
        "total_matrix_entries": total_entries,
    }
