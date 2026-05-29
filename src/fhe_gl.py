from __future__ import annotations

import numpy as np

from .logging_utils import ct_info, elapsed, now_seconds


def gl_slots(engine) -> int:
    return int(np.prod(getattr(engine, "shape")))


def flat_to_shape(vec: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(int(np.prod(shape)), dtype=np.float64)
    out[: vec.size] = vec.ravel()
    return out.reshape(shape)


def check_plain_weight_matrix_multiply(engine, sk, mm_key) -> dict:
    shape = tuple(getattr(engine, "shape"))
    ct = engine.encrypt(flat_to_shape(np.arange(64, dtype=np.float64), shape), sk)
    tests = {}
    for name, matrix in {
        "numpy_16x64_times_ct": np.ones((16, 64), dtype=np.float64),
        "numpy_full_shape_times_ct": np.ones(shape, dtype=np.float64),
        "glplaintext_full_shape_times_ct": engine.encode(np.ones(shape, dtype=np.float64)),
    }.items():
        try:
            out = engine.matrix_multiply(matrix, ct, mm_key)
            tests[name] = {"ok": True, **ct_info(out)}
        except Exception as exc:
            tests[name] = {"ok": False, "exception_type": type(exc).__name__, "exception": str(exc)}
    return tests


def run_ct_ct_microbenchmark(engine, sk, mm_key) -> dict:
    shape = tuple(getattr(engine, "shape"))
    a = engine.encrypt(np.ones(shape, dtype=np.float64), sk)
    b = engine.encrypt(np.eye(shape[-2], shape[-1], dtype=np.float64), sk)
    t = now_seconds()
    out = engine.matrix_multiply(a, b, mm_key)
    return {"ok": True, "matrix_multiply_s": elapsed(t), "output": ct_info(out)}
