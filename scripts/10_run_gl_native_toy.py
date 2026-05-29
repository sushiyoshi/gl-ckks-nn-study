from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_layout import make_gl_bias_column, make_gl_column, make_gl_weight_matrix, read_gl_column
from src.logging_utils import RESULTS, SEED, ct_info, elapsed, now_seconds, write_csv, write_json
from src.metrics import argmax_agreement, error_metrics
from src.overhead_metrics import overhead_record
from src.polynomial import eval_power_polynomial, fit_relu_power_polynomial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    args = parser.parse_args()

    from desilofhe import GLEngine

    rng = np.random.default_rng(SEED)
    x = rng.normal(0.0, 0.5, size=32)
    w1 = rng.normal(0.0, 0.12, size=(32, 32))
    b1 = rng.normal(0.0, 0.03, size=32)
    w2 = rng.normal(0.0, 0.12, size=(32, 32))
    b2 = rng.normal(0.0, 0.03, size=32)
    coeffs = fit_relu_power_polynomial(args.degree, (-3.0, 3.0))
    z1 = w1 @ x + b1
    h1 = eval_power_polynomial(z1, coeffs)
    plain = w2 @ h1 + b2

    engine = GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    slot_count = int(np.prod(shape))
    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    had_key = engine.create_hadamard_multiplication_key(sk)

    level_log = []
    timing_log = {}
    t = now_seconds()
    ct = engine.encrypt(make_gl_column(x, shape), sk)
    timing_log["encrypt_s"] = elapsed(t)
    level_log.append({"stage": "input", **ct_info(ct)})

    t = now_seconds()
    ct = engine.matrix_multiply(engine.encode(make_gl_weight_matrix(w1, shape)), ct, mm_key)
    ct = engine.add(ct, engine.encode(make_gl_bias_column(b1, shape)))
    timing_log["linear1_add_s"] = elapsed(t)
    level_log.append({"stage": "linear1", **ct_info(ct)})

    t = now_seconds()
    ct = engine.evaluate_polynomial(ct, coeffs.astype(np.float64), had_key)
    timing_log["activation_s"] = elapsed(t)
    level_log.append({"stage": "activation", **ct_info(ct)})

    t = now_seconds()
    ct = engine.matrix_multiply(engine.encode(make_gl_weight_matrix(w2, shape)), ct, mm_key)
    ct = engine.add(ct, engine.encode(make_gl_bias_column(b2, shape)))
    timing_log["linear2_add_s"] = elapsed(t)
    level_log.append({"stage": "linear2", **ct_info(ct)})

    t = now_seconds()
    dec = np.asarray(engine.decrypt(ct, sk), dtype=np.float64)
    timing_log["decrypt_s"] = elapsed(t)
    out = read_gl_column(dec, 32)
    err = error_metrics(plain.reshape(1, -1), out.reshape(1, -1), "logits")
    payload = {
        "ok": True,
        "semantic_validation_passed": bool(err["logits_relative_l2"] < 1e-5),
        "shape": shape,
        "logical_dims": [32, 32, 32],
        "physical_dim": 32,
        "slot_count": slot_count,
        **overhead_record((32, 32, 32), 32, 32, slot_count, 2, 2, args.degree),
        "operation_counts": {"matrix_multiply": 2, "add": 2, "polynomial": 1},
        "level_log": level_log,
        "timing_log": timing_log,
        "argmax_agreement": argmax_agreement(plain.reshape(1, -1), out.reshape(1, -1)),
        "error_metrics": err,
    }
    write_json(RESULTS / "gl_native_toy.json", payload)
    write_csv(RESULTS / "gl_native_toy.csv", [{k: v for k, v in payload.items() if not isinstance(v, (dict, list, tuple))}])
    print(RESULTS / "gl_native_toy.json")


if __name__ == "__main__":
    main()
