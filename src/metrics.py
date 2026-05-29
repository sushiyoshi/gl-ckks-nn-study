from __future__ import annotations

import numpy as np


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=1) == y))


def argmax_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1)))


def error_metrics(reference: np.ndarray, actual: np.ndarray, prefix: str) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    diff = actual - reference
    denom = float(np.linalg.norm(reference.ravel()))
    return {
        f"{prefix}_linf": float(np.max(np.abs(diff))),
        f"{prefix}_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_relative_l2": float(np.linalg.norm(diff.ravel()) / (denom + 1e-12)),
    }
