from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_packing import broadcast_bias_to_columns, broadcast_weight_to_batches, pack_columns, packing_stats, unpack_logits
from src.logging_utils import RESULTS, ct_info, elapsed, now_seconds, write_csv, write_json
from src.metrics import accuracy, argmax_agreement, error_metrics
from src.model_pca32 import load_pca32_model, require_sample_count
from src.polynomial import fit_relu_power_polynomial


def save_outputs(base: str, n_samples: int, meta: dict, rows: list[dict]) -> None:
    write_json(RESULTS / f"{base}.json", meta)
    write_csv(RESULTS / f"{base}.csv", rows)
    write_json(RESULTS / f"{base}_n{n_samples}.json", meta)
    write_csv(RESULTS / f"{base}_n{n_samples}.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=32)
    args = parser.parse_args()

    from desilofhe import GLEngine

    model, arrays = load_pca32_model()
    require_sample_count(args.n_samples, arrays["x_test_32"].shape[0], label="x_test_32")
    n = args.n_samples
    x = arrays["x_test_32"][:n]
    y = arrays["y_test"][:n]
    z_train = model.relu_forward(arrays["x_train_32"])["z1"]
    radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z_train), 0.995)))))
    coeffs = fit_relu_power_polynomial(args.degree, (-radius, radius)).astype(np.float64)
    relu_plain = model.relu_forward(x)["logits"]
    poly_plain = model.poly_forward(x, coeffs)["logits"]

    t_total = now_seconds()
    engine = GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    had_key = engine.create_hadamard_multiplication_key(sk)
    input_tensor, layout = pack_columns(x, shape)
    used_batches = layout["used_batches"]

    w1_pt = engine.encode(broadcast_weight_to_batches(model.w1.T, shape, used_batches, 32, 32))
    w2_pt = engine.encode(broadcast_weight_to_batches(model.w2.T, shape, used_batches, 10, 32))
    b1_pt = engine.encode(broadcast_bias_to_columns(model.b1, shape, layout, 32))
    b2_pt = engine.encode(broadcast_bias_to_columns(model.b2, shape, layout, 10))

    timing = {}
    level_log = []
    t = now_seconds()
    ct = engine.encrypt(input_tensor, sk)
    timing["encrypt_s"] = elapsed(t)
    level_log.append({"stage": "input", **ct_info(ct)})
    t = now_seconds()
    ct = engine.matrix_multiply(w1_pt, ct, mm_key)
    ct = engine.add(ct, b1_pt)
    timing["linear1_add_s"] = elapsed(t)
    level_log.append({"stage": "linear1", **ct_info(ct)})
    t = now_seconds()
    ct = engine.evaluate_polynomial(ct, coeffs, had_key)
    timing["activation_s"] = elapsed(t)
    level_log.append({"stage": "activation", **ct_info(ct)})
    t = now_seconds()
    ct = engine.matrix_multiply(w2_pt, ct, mm_key)
    ct = engine.add(ct, b2_pt)
    timing["linear2_add_s"] = elapsed(t)
    level_log.append({"stage": "linear2", **ct_info(ct)})
    t = now_seconds()
    dec = np.asarray(engine.decrypt(ct, sk), dtype=np.float64)
    timing["decrypt_s"] = elapsed(t)
    logits = unpack_logits(dec, layout, 10)
    runtime_total = elapsed(t_total)
    err = error_metrics(poly_plain, logits, "logits")
    rows = [{"sample_index": i, "true_label": int(y[i]), "relu_pred": int(np.argmax(relu_plain[i])), "poly_pred": int(np.argmax(poly_plain[i])), "gl_pred": int(np.argmax(logits[i]))} for i in range(n)]
    meta = {
        "ok": True,
        "semantic_validation_passed": bool(err["logits_relative_l2"] < 1e-5),
        "shape": shape,
        "logical_dims": [32, 32, 10],
        "physical_dim": 32,
        "slot_count": int(np.prod(shape)),
        "accuracy_relu_plain": accuracy(relu_plain, y),
        "accuracy_poly_plain": accuracy(poly_plain, y),
        "accuracy_gl_decrypted": accuracy(logits, y),
        "argmax_agreement_poly_vs_gl": argmax_agreement(poly_plain, logits),
        "error_metrics": err,
        "operation_counts": {"matrix_multiply": 2, "add": 2, "polynomial": 1},
        "polynomial_degree": args.degree,
        "used_batches": used_batches,
        "used_columns_last_batch": layout["used_columns_last_batch"],
        "output_padding_ratio": 22 / 32,
        **packing_stats(n, 32, shape),
        "level_log": level_log,
        "timing_log": timing,
        "runtime_total_s": runtime_total,
        "runtime_per_sample_amortized_s": runtime_total / n,
    }
    save_outputs("gl_padded_pca32_packed_results", n, meta, rows)
    print(RESULTS / f"gl_padded_pca32_packed_results_n{n}.json")


if __name__ == "__main__":
    main()
