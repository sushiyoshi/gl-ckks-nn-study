from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_encrypted_weight import classify_failure, scalar_metrics
from src.logging_utils import RESULTS, SEED, ct_info, elapsed, now_seconds, write_json


def run_case(engine: Any, sk: Any, mm_key: Any, name: str, left: Any, right: Any, expected: np.ndarray) -> dict[str, Any]:
    t = now_seconds()
    try:
        out = engine.matrix_multiply(left, right, mm_key)
        mm_s = elapsed(t)
        t = now_seconds()
        dec = np.asarray(engine.decrypt(out, sk), dtype=np.float64)
        decrypt_s = elapsed(t)
        metrics = scalar_metrics(expected, dec, "matrix")
        return {
            "ok": True,
            "case": name,
            "semantic_validation_passed": bool(metrics["matrix_relative_l2"] < 1e-5 and metrics["matrix_allclose"]),
            "failure_category": None,
            "matrix_multiply_s": mm_s,
            "decryption_s": decrypt_s,
            "output": ct_info(out),
            **metrics,
        }
    except Exception as exc:
        return {
            "ok": False,
            "case": name,
            "semantic_validation_passed": False,
            "failure_category": classify_failure(exc),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "matrix_multiply_s": elapsed(t),
        }


def main() -> None:
    from desilofhe import GLEngine

    rng = np.random.default_rng(SEED)
    t_total = now_seconds()
    engine = GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    t = now_seconds()
    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    key_generation_s = elapsed(t)

    a = np.zeros(shape, dtype=np.float64)
    b = np.zeros(shape, dtype=np.float64)
    for batch in (0, 1):
        a[batch] = rng.normal(0.0, 0.2, size=shape[1:])
        b[batch] = rng.normal(0.0, 0.2, size=shape[1:])
    expected = np.matmul(a, b)

    t = now_seconds()
    ct_a = engine.encrypt(a, sk)
    ct_b = engine.encrypt(b, sk)
    encryption_s = elapsed(t)
    a_pt = engine.encode(a)
    b_pt = engine.encode(b)

    cases = [
        run_case(engine, sk, mm_key, "ct_ct", ct_a, ct_b, expected),
        run_case(engine, sk, mm_key, "pt_ct", a_pt, ct_b, expected),
        run_case(engine, sk, mm_key, "ct_pt", ct_a, b_pt, expected),
    ]
    payload = {
        "ok": any(case["ok"] for case in cases),
        "semantic_validation_passed": all(case["semantic_validation_passed"] for case in cases),
        "shape": shape,
        "tested_batches": [0, 1],
        "key_generation_s": key_generation_s,
        "encryption_s": encryption_s,
        "runtime_total_s": elapsed(t_total),
        "inputs": {"ct_a": ct_info(ct_a), "ct_b": ct_info(ct_b)},
        "cases": cases,
    }
    write_json(RESULTS / "gl_ctct_matrix_multiply_probe.json", payload)
    print(RESULTS / "gl_ctct_matrix_multiply_probe.json")


if __name__ == "__main__":
    main()
