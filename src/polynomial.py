from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def fit_relu_power_polynomial(degree: int, interval: tuple[float, float], n_grid: int = 4001) -> np.ndarray:
    lo, hi = interval
    xs = np.linspace(lo, hi, n_grid, dtype=np.float64)
    ys = relu(xs)
    # np.polyfit returns descending powers; FHE APIs here expect ascending powers.
    return np.polyfit(xs, ys, degree).astype(np.float64)[::-1]


def eval_power_polynomial(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for c in coeffs[::-1]:
        out = out * x + c
    return out


def choose_interval(pre_activation: np.ndarray, fallback: tuple[float, float] = (-3.0, 3.0)) -> tuple[float, float]:
    q = float(np.quantile(np.abs(pre_activation), 0.995))
    radius = max(abs(fallback[0]), abs(fallback[1]), min(8.0, np.ceil(q)))
    return (-radius, radius)


def out_of_interval_fraction(x: np.ndarray, interval: tuple[float, float]) -> float:
    lo, hi = interval
    return float(np.mean((x < lo) | (x > hi)))
