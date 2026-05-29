from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_layout import make_gl_bias_column, make_gl_column, make_gl_weight_matrix, read_gl_column
from src.logging_utils import RESULTS, ct_info, elapsed, now_seconds, write_csv, write_json
from src.metrics import accuracy, argmax_agreement, error_metrics
from src.model_pca32 import load_pca32_model
from src.overhead_metrics import overhead_record
from src.polynomial import fit_relu_power_polynomial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=1)
    args = parser.parse_args()

    from desilofhe import GLEngine

    model, arrays = load_pca32_model()
    z_train = model.relu_forward(arrays["x_train_32"])["z1"]
    radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z_train), 0.995)))))
    coeffs = fit_relu_power_polynomial(args.degree, (-radius, radius))
    x = arrays["x_test_32"][: args.n_samples]
    y = arrays["y_test"][: args.n_samples]
    relu_plain = model.relu_forward(x)["logits"]
    poly_plain = model.poly_forward(x, coeffs)["logits"]

    engine = GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    slot_count = int(np.prod(shape))
    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    had_key = engine.create_hadamard_multiplication_key(sk)
    w1_pt = engine.encode(make_gl_weight_matrix(model.w1.T, shape))
    w2_pt = engine.encode(make_gl_weight_matrix(model.w2.T, shape))
    b1_pt = engine.encode(make_gl_bias_column(model.b1, shape))
    b2_pt = engine.encode(make_gl_bias_column(model.b2, shape))

    rows = []
    outs = []
    level_log = []
    timing_total = {"encrypt_s": 0.0, "linear1_add_s": 0.0, "activation_s": 0.0, "linear2_add_s": 0.0, "decrypt_s": 0.0}
    for i, sample in enumerate(x):
        t = now_seconds()
        ct = engine.encrypt(make_gl_column(sample, shape), sk)
        timing_total["encrypt_s"] += elapsed(t)
        t = now_seconds()
        ct = engine.matrix_multiply(w1_pt, ct, mm_key)
        ct = engine.add(ct, b1_pt)
        timing_total["linear1_add_s"] += elapsed(t)
        l1 = ct_info(ct)
        t = now_seconds()
        ct = engine.evaluate_polynomial(ct, coeffs.astype(np.float64), had_key)
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
        rows.append({"sample_index": i, "true_label": int(y[i]), "relu_pred": int(np.argmax(relu_plain[i])), "poly_pred": int(np.argmax(poly_plain[i])), "gl_pred": int(np.argmax(logits)), "linear1_level": l1["level"], "activation_level": lact["level"], "linear2_level": l2["level"]})
        if i == 0:
            level_log = [{"stage": "linear1", **l1}, {"stage": "activation", **lact}, {"stage": "linear2", **l2}]

    gl = np.vstack(outs)
    err = error_metrics(poly_plain[: len(gl)], gl, "logits")
    meta = {
        "ok": True,
        "semantic_validation_passed": bool(err["logits_relative_l2"] < 1e-5),
        "shape": shape,
        "logical_dims": [32, 32, 10],
        "physical_dim": 32,
        "slot_count": slot_count,
        "accuracy_relu_plain": accuracy(relu_plain, y),
        "accuracy_poly_plain": accuracy(poly_plain, y),
        "accuracy_gl_decrypted": accuracy(gl, y[: len(gl)]),
        "argmax_agreement_poly_vs_gl": argmax_agreement(poly_plain[: len(gl)], gl),
        "error_metrics": err,
        "padded_output_dim": 22,
        "output_padding_ratio": 22 / 32,
        **overhead_record((32, 32, 10), 32, 32, slot_count, 2, 2, args.degree),
        "operation_counts": {"matrix_multiply": 2, "add": 2, "polynomial": 1},
        "level_log": level_log,
        "timing_log": timing_total,
    }
    write_json(RESULTS / "gl_padded_pca32_results.json", meta)
    write_csv(RESULTS / "gl_padded_pca32_results.csv", rows)
    print(RESULTS / "gl_padded_pca32_results.json")


if __name__ == "__main__":
    main()
