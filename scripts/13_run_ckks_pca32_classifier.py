from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, ct_info, elapsed, now_seconds, write_csv, write_json
from src.metrics import accuracy, argmax_agreement, error_metrics
from src.model_pca32 import load_pca32_model, require_sample_count
from src.polynomial import fit_relu_power_polynomial


def linear_matrix(w_in_out: np.ndarray, slots: int) -> np.ndarray:
    in_dim, out_dim = w_in_out.shape
    mat = np.zeros((slots, slots), dtype=np.float64)
    mat[:out_dim, :in_dim] = w_in_out.T
    return mat


def bias_vector(bias: np.ndarray, slots: int) -> np.ndarray:
    out = np.zeros(slots, dtype=np.float64)
    out[: bias.size] = bias
    return out


def packed(vec: np.ndarray, slots: int) -> np.ndarray:
    out = np.zeros(slots, dtype=np.float64)
    out[: vec.size] = vec
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=1)
    args = parser.parse_args()

    from desilofhe import Engine

    model, arrays = load_pca32_model()
    require_sample_count(args.n_samples, arrays["x_test_32"].shape[0], label="x_test_32")
    z_train = model.relu_forward(arrays["x_train_32"])["z1"]
    radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z_train), 0.995)))))
    coeffs = fit_relu_power_polynomial(args.degree, (-radius, radius))
    x = arrays["x_test_32"][: args.n_samples]
    y = arrays["y_test"][: args.n_samples]
    poly_plain = model.poly_forward(x, coeffs)["logits"]

    engine = Engine()
    slots = int(engine.slot_count)
    sk = engine.create_secret_key()
    rot_key = engine.create_rotation_key(sk)
    relin_key = engine.create_relinearization_key(sk)
    m1 = linear_matrix(model.w1, slots)
    m2 = linear_matrix(model.w2, slots)
    b1 = bias_vector(model.b1, slots)
    b2 = bias_vector(model.b2, slots)

    rows = []
    outs = []
    level_log = []
    timing_total = {"encrypt_s": 0.0, "linear1_add_s": 0.0, "activation_s": 0.0, "linear2_add_s": 0.0, "decrypt_s": 0.0}
    for i, sample in enumerate(x):
        t = now_seconds()
        ct = engine.encrypt(packed(sample, slots), sk)
        timing_total["encrypt_s"] += elapsed(t)
        t = now_seconds()
        ct = engine.multiply_matrix(m1, ct, rot_key)
        ct = engine.add(ct, b1)
        timing_total["linear1_add_s"] += elapsed(t)
        l1 = ct_info(ct)
        t = now_seconds()
        ct = engine.evaluate_polynomial(ct, [float(c) for c in coeffs], relin_key)
        timing_total["activation_s"] += elapsed(t)
        lact = ct_info(ct)
        t = now_seconds()
        ct = engine.multiply_matrix(m2, ct, rot_key)
        ct = engine.add(ct, b2)
        timing_total["linear2_add_s"] += elapsed(t)
        l2 = ct_info(ct)
        t = now_seconds()
        dec = np.asarray(engine.decrypt(ct, sk), dtype=np.float64)
        timing_total["decrypt_s"] += elapsed(t)
        logits = dec[:10]
        outs.append(logits)
        rows.append({"sample_index": i, "true_label": int(y[i]), "poly_pred": int(np.argmax(poly_plain[i])), "ckks_pred": int(np.argmax(logits)), "linear1_level": l1["level"], "activation_level": lact["level"], "linear2_level": l2["level"]})
        if i == 0:
            level_log = [{"stage": "linear1", **l1}, {"stage": "activation", **lact}, {"stage": "linear2", **l2}]

    ckks = np.vstack(outs)
    err = error_metrics(poly_plain[: len(ckks)], ckks, "logits")
    meta = {
        "ok": True,
        "semantic_validation_passed": bool(err["logits_relative_l2"] < 1e-2),
        "shape": [slots],
        "logical_dims": [32, 32, 10],
        "physical_dim": slots,
        "slot_count": slots,
        "used_slots": 32,
        "packing_utilization": 32 / slots,
        "polynomial_degree": args.degree,
        "accuracy_poly_plain": accuracy(poly_plain, y),
        "accuracy_ckks_decrypted": accuracy(ckks, y[: len(ckks)]),
        "argmax_agreement_poly_vs_ckks": argmax_agreement(poly_plain[: len(ckks)], ckks),
        "error_metrics": err,
        "operation_counts": {"matrix_multiply": 2, "add": 2, "polynomial": 1},
        "level_log": level_log,
        "timing_log": timing_total,
    }
    write_json(RESULTS / "ckks_pca32_results.json", meta)
    write_csv(RESULTS / "ckks_pca32_results.csv", rows)
    print(RESULTS / "ckks_pca32_results.json")


if __name__ == "__main__":
    main()
