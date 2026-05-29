from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_layout import make_gl_bias_column, make_gl_column, make_gl_weight_matrix, read_gl_column
from src.logging_utils import DATA, RESULTS, ct_info, elapsed, now_seconds, write_csv, write_json
from src.metrics import argmax_agreement, error_metrics
from src.model import load_model
from src.overhead_metrics import overhead_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=1)
    args = parser.parse_args()

    from desilofhe import GLEngine

    model, arrays = load_model()
    poly = np.load(DATA / f"relu_poly_degree{args.degree}.npz")
    coeffs = poly["coeffs"].astype(np.float64)
    x = arrays["x_test_scaled"][: args.n_samples]
    y = arrays["y_test"][: args.n_samples]
    poly_plain = model.poly_forward(x, coeffs)["logits"]

    engine = GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    slot_count = int(np.prod(shape))
    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    had_key = engine.create_hadamard_multiplication_key(sk)

    w1_out_in = model.w1.T
    w10_pt = engine.encode(make_gl_weight_matrix(w1_out_in[:, :32], shape))
    w11_pt = engine.encode(make_gl_weight_matrix(w1_out_in[:, 32:64], shape))
    b1_pt = engine.encode(make_gl_bias_column(model.b1, shape))
    w2_pt = engine.encode(make_gl_weight_matrix(model.w2.T, shape))
    b2_pt = engine.encode(make_gl_bias_column(model.b2, shape))

    rows = []
    outs = []
    level_log = []
    timing_total = {"encrypt_s": 0.0, "linear1_blocks_s": 0.0, "activation_s": 0.0, "linear2_add_s": 0.0, "decrypt_s": 0.0}
    for i, sample in enumerate(x):
        t = now_seconds()
        ct0 = engine.encrypt(make_gl_column(sample[:32], shape), sk)
        ct1 = engine.encrypt(make_gl_column(sample[32:64], shape), sk)
        timing_total["encrypt_s"] += elapsed(t)

        t = now_seconds()
        h0 = engine.matrix_multiply(w10_pt, ct0, mm_key)
        h1 = engine.matrix_multiply(w11_pt, ct1, mm_key)
        ct = engine.add(h0, h1)
        ct = engine.add(ct, b1_pt)
        timing_total["linear1_blocks_s"] += elapsed(t)
        l1 = ct_info(ct)

        t = now_seconds()
        ct = engine.evaluate_polynomial(ct, coeffs, had_key)
        timing_total["activation_s"] += elapsed(t)
        lact = ct_info(ct)

        t = now_seconds()
        ct = engine.matrix_multiply(w2_pt, ct, mm_key)
        ct = engine.add(ct, b2_pt)
        timing_total["linear2_add_s"] += elapsed(t)
        l2 = ct_info(ct)

        t = now_seconds()
        dec = np.asarray(engine.decrypt(ct, sk), dtype=np.float64)
        timing_total["decrypt_s"] += elapsed(t)
        logits = read_gl_column(dec, 10)
        outs.append(logits)
        rows.append({"sample_index": i, "true_label": int(y[i]), "poly_pred": int(np.argmax(poly_plain[i])), "gl_pred": int(np.argmax(logits)), "linear1_level": l1["level"], "activation_level": lact["level"], "linear2_level": l2["level"]})
        if i == 0:
            level_log = [{"stage": "linear1_block_sum", **l1}, {"stage": "activation", **lact}, {"stage": "linear2", **l2}]

    gl = np.vstack(outs)
    err = error_metrics(poly_plain[: len(gl)], gl, "logits")
    meta = {
        "ok": True,
        "semantic_validation_passed": bool(err["logits_relative_l2"] < 1e-5),
        "shape": shape,
        "logical_dims": [64, 16, 10],
        "physical_dim": 32,
        "slot_count": slot_count,
        "argmax_agreement_poly_vs_gl": argmax_agreement(poly_plain[: len(gl)], gl),
        "error_metrics": err,
        "hidden_padding_ratio": 16 / 32,
        "input_block_count": 2,
        "block_decomposition_overhead": {"64_to_16_matrix_multiply_count": 2, "ideal_unblocked_count": 1, "ratio": 2.0},
        **overhead_record((64, 16, 10), 32, 64, slot_count, 3, 3, args.degree, mask_count=0, block_count=2),
        "operation_counts": {"matrix_multiply": 3, "add": 3, "polynomial": 1, "mask": 0},
        "level_log": level_log,
        "timing_log": timing_total,
    }
    write_json(RESULTS / "gl_blocked_digits64_results.json", meta)
    write_csv(RESULTS / "gl_blocked_digits64_results.csv", rows)
    print(RESULTS / "gl_blocked_digits64_results.json")


if __name__ == "__main__":
    main()
