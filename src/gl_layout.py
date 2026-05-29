from __future__ import annotations

import numpy as np


def make_gl_column(vec: np.ndarray, shape: tuple[int, int, int], batch: int = 0, col: int = 0) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64).ravel()
    if v.size > shape[1]:
        raise ValueError(f"vector length {v.size} exceeds GL physical rows {shape[1]}")
    arr[batch, : v.size, col] = v
    return arr


def make_gl_weight_matrix(w_out_in: np.ndarray, shape: tuple[int, int, int], batch: int = 0) -> np.ndarray:
    w = np.asarray(w_out_in, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape {w.shape}")
    out_dim, in_dim = w.shape
    if out_dim > shape[1] or in_dim > shape[2]:
        raise ValueError(f"weight shape {w.shape} exceeds GL physical matrix {shape[1:]}")
    arr = np.zeros(shape, dtype=np.float64)
    arr[batch, :out_dim, :in_dim] = w
    return arr


def make_gl_bias_column(bias: np.ndarray, shape: tuple[int, int, int], batch: int = 0) -> np.ndarray:
    return make_gl_column(np.asarray(bias, dtype=np.float64), shape, batch=batch, col=0)


def read_gl_column(dec: np.ndarray, n: int, batch: int = 0, col: int = 0) -> np.ndarray:
    return np.asarray(dec, dtype=np.float64)[batch, :n, col].copy()


def packing_utilization(logical_rows: int, logical_cols: int, shape: tuple[int, int, int]) -> float:
    return float((logical_rows * logical_cols) / np.prod(shape))


def padding_stats(logical_in: int, logical_out: int, physical_dim: int = 32) -> dict:
    physical_entries = physical_dim * physical_dim
    used_entries = logical_in * logical_out
    return {
        "logical_in": logical_in,
        "logical_out": logical_out,
        "physical_dim": physical_dim,
        "used_entries": used_entries,
        "physical_entries": physical_entries,
        "unused_entries": physical_entries - used_entries,
        "utilization": used_entries / physical_entries,
        "padding_ratio": (physical_entries - used_entries) / physical_entries,
    }
