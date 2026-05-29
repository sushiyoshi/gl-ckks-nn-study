from __future__ import annotations

import math
from typing import Any

import numpy as np


def pack_samples_blocks(samples: np.ndarray, block_dim: int, slot_count: int, chunk_samples: int | None = None) -> tuple[list[np.ndarray], dict[str, Any]]:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"samples must be 2D, got {x.shape}")
    n_samples, feature_dim = x.shape
    if feature_dim > block_dim:
        raise ValueError(f"feature_dim {feature_dim} exceeds block_dim {block_dim}")
    max_samples_per_ct = slot_count // block_dim
    samples_per_ct = max_samples_per_ct if chunk_samples is None else min(chunk_samples, max_samples_per_ct)
    if samples_per_ct <= 0:
        raise ValueError("samples_per_ct must be positive")
    chunks = []
    placements = []
    for chunk_start in range(0, n_samples, samples_per_ct):
        chunk = np.zeros(slot_count, dtype=np.float64)
        chunk_index = len(chunks)
        part = x[chunk_start : chunk_start + samples_per_ct]
        for local_i, sample in enumerate(part):
            sample_index = chunk_start + local_i
            start = local_i * block_dim
            chunk[start : start + feature_dim] = sample
            placements.append({"sample_index": sample_index, "chunk": chunk_index, "local_index": local_i, "slot_start": start})
        chunks.append(chunk)
    layout = {
        "n_samples": n_samples,
        "feature_dim": feature_dim,
        "block_dim": block_dim,
        "slot_count": slot_count,
        "samples_per_ct": samples_per_ct,
        "max_samples_per_ct": max_samples_per_ct,
        "ciphertext_count": len(chunks),
        "placements": placements,
    }
    return chunks, layout


def unpack_logits_blocks(decoded_chunks: list[np.ndarray], layout: dict[str, Any], n_logits: int) -> np.ndarray:
    out = np.zeros((layout["n_samples"], n_logits), dtype=np.float64)
    block_dim = layout["block_dim"]
    for item in layout["placements"]:
        dec = np.asarray(decoded_chunks[item["chunk"]], dtype=np.float64)
        start = item["local_index"] * block_dim
        out[item["sample_index"]] = dec[start : start + n_logits]
    return out


def make_block_diagonal_linear_matrix(
    w_left: np.ndarray,
    block_dim: int,
    samples_per_ct: int,
    slot_count: int,
    out_rows: int | None = None,
    in_rows: int | None = None,
) -> np.ndarray:
    w = np.asarray(w_left, dtype=np.float64)
    out = w.shape[0] if out_rows is None else out_rows
    inn = w.shape[1] if in_rows is None else in_rows
    if out > block_dim or inn > block_dim:
        raise ValueError(f"logical block {(out, inn)} exceeds block_dim {block_dim}")
    matrix = np.zeros((slot_count, slot_count), dtype=np.float64)
    max_blocks = slot_count // block_dim
    blocks = min(samples_per_ct, max_blocks)
    for i in range(blocks):
        start = i * block_dim
        matrix[start : start + out, start : start + inn] = w[:out, :inn]
    return matrix


def make_repeated_bias_vector(
    bias: np.ndarray,
    block_dim: int,
    samples_per_ct: int,
    slot_count: int,
    active_rows: int,
) -> np.ndarray:
    b = np.asarray(bias, dtype=np.float64).ravel()
    vec = np.zeros(slot_count, dtype=np.float64)
    max_blocks = slot_count // block_dim
    blocks = min(samples_per_ct, max_blocks)
    for i in range(blocks):
        start = i * block_dim
        vec[start : start + active_rows] = b[:active_rows]
    return vec


def packing_stats(n_samples: int, block_dim: int, slot_count: int, ciphertext_count: int) -> dict[str, Any]:
    samples_per_ct = slot_count // block_dim
    capacity = ciphertext_count * samples_per_ct
    used_slots = n_samples * block_dim
    total_slots = ciphertext_count * slot_count
    return {
        "n_samples": n_samples,
        "block_dim": block_dim,
        "slot_count": slot_count,
        "samples_per_ciphertext": samples_per_ct,
        "ciphertext_count": ciphertext_count,
        "sample_packing_utilization": n_samples / capacity if capacity else 0.0,
        "used_slots": used_slots,
        "total_slots_across_ciphertexts": total_slots,
        "slot_utilization": used_slots / total_slots if total_slots else 0.0,
    }


def dense_matrix_memory_stats(*matrices: np.ndarray) -> dict[str, Any]:
    total = sum(int(m.nbytes) for m in matrices)
    return {
        "dense_matrix_count": len(matrices),
        "dense_matrix_nbytes_total": total,
        "dense_matrix_mib_total": total / (1024 * 1024),
        "dense_matrix_shapes": [list(m.shape) for m in matrices],
        "dense_matrix_dtypes": [str(m.dtype) for m in matrices],
    }
