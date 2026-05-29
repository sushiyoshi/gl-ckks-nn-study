from __future__ import annotations

import numpy as np

from .logging_utils import ct_info, elapsed, now_seconds


def padded_linear_matrix(weight_in_out: np.ndarray, bias: np.ndarray, slots: int) -> np.ndarray:
    in_dim, out_dim = weight_in_out.shape
    if max(in_dim, out_dim) > slots:
        raise ValueError(f"layer dims {(in_dim, out_dim)} exceed CKKS slots {slots}")
    matrix = np.zeros((slots, slots), dtype=np.float64)
    # Plain model uses x @ W + b. CKKS multiply_matrix computes M @ x, so use W.T.
    matrix[:out_dim, :in_dim] = weight_in_out.T
    matrix[:out_dim, slots - 1] = bias
    return matrix


def packed_input(x: np.ndarray, slots: int) -> np.ndarray:
    arr = np.zeros(slots, dtype=np.float64)
    arr[: x.shape[0]] = x
    arr[slots - 1] = 1.0
    return arr


def run_one(engine, sk, rot_key, relin_key, x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, coeffs: np.ndarray) -> dict:
    slots = int(getattr(engine, "slot_count", 8192))
    log = {}
    t = now_seconds()
    ct = engine.encrypt(packed_input(x, slots), sk)
    log["encrypt_s"] = elapsed(t)
    log["input"] = ct_info(ct)

    m1 = padded_linear_matrix(w1, b1, slots)
    t = now_seconds()
    ct = engine.multiply_matrix(m1, ct, rot_key)
    log["linear1_s"] = elapsed(t)
    log["linear1"] = ct_info(ct)

    t = now_seconds()
    ct = engine.evaluate_polynomial(ct, [float(c) for c in coeffs], relin_key)
    log["activation_s"] = elapsed(t)
    log["activation"] = ct_info(ct)

    m2 = padded_linear_matrix(w2, b2, slots)
    t = now_seconds()
    ct = engine.multiply_matrix(m2, ct, rot_key)
    log["linear2_s"] = elapsed(t)
    log["linear2"] = ct_info(ct)

    t = now_seconds()
    dec = np.asarray(engine.decrypt(ct, sk), dtype=np.float64)
    log["decrypt_s"] = elapsed(t)
    return {"logits": dec[: b2.shape[0]], "log": log}
