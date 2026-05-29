from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_packing import broadcast_bias_to_columns, broadcast_weight_to_batches, packing_stats, split64_to_two_packed_tensors, unpack_logits
from src.logging_utils import DATA, RESULTS, ct_info, elapsed, now_seconds, write_csv, write_json
from src.metrics import accuracy, argmax_agreement, error_metrics
from src.model import load_model


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

    model, arrays = load_model()
    n = min(args.n_samples, arrays["x_test_scaled"].shape[0])
    x = arrays["x_test_scaled"][:n]
    y = arrays["y_test"][:n]
    coeffs = np.load(DATA / f"relu_poly_degree{args.degree}.npz")["coeffs"].astype(np.float64)
    poly_plain = model.poly_forward(x, coeffs)["logits"]

    t_total = now_seconds()
    engine = GLEngine()
    shape = tuple(int(v) for v in engine.shape)
    sk = engine.create_secret_key()
    mm_key = engine.create_matrix_multiplication_key(sk)
    had_key = engine.create_hadamard_multiplication_key(sk)
    x0_tensor, x1_tensor, layout = split64_to_two_packed_tensors(x, shape)
    used_batches = layout["used_batches"]
    w1 = model.w1.T
    w10_pt = engine.encode(broadcast_weight_to_batches(w1[:, :32], shape, used_batches, 16, 32))
    w11_pt = engine.encode(broadcast_weight_to_batches(w1[:, 32:64], shape, used_batches, 16, 32))
    w2_pt = engine.encode(broadcast_weight_to_batches(model.w2.T, shape, used_batches, 10, 16))
    b1_pt = engine.encode(broadcast_bias_to_columns(model.b1, shape, layout, 16))
    b2_pt = engine.encode(broadcast_bias_to_columns(model.b2, shape, layout, 10))

    timing = {}
    level_log = []
    t = now_seconds()
    ct0 = engine.encrypt(x0_tensor, sk)
    ct1 = engine.encrypt(x1_tensor, sk)
    timing["encrypt_s"] = elapsed(t)
    level_log.append({"stage": "input0", **ct_info(ct0)})
    level_log.append({"stage": "input1", **ct_info(ct1)})
    t = now_seconds()
    h0 = engine.matrix_multiply(w10_pt, ct0, mm_key)
    h1 = engine.matrix_multiply(w11_pt, ct1, mm_key)
    ct = engine.add(h0, h1)
    ct = engine.add(ct, b1_pt)
    timing["linear1_blocks_s"] = elapsed(t)
    level_log.append({"stage": "linear1_block_sum", **ct_info(ct)})
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
    rows = [{"sample_index": i, "true_label": int(y[i]), "poly_pred": int(np.argmax(poly_plain[i])), "gl_pred": int(np.argmax(logits[i]))} for i in range(n)]
    meta = {
        "ok": True,
        "semantic_validation_passed": bool(err["logits_relative_l2"] < 1e-5),
        "shape": shape,
        "logical_dims": [64, 16, 10],
        "physical_dim": 32,
        "slot_count": int(np.prod(shape)),
        "accuracy_poly_plain": accuracy(poly_plain, y),
        "accuracy_gl_decrypted": accuracy(logits, y),
        "argmax_agreement_poly_vs_gl": argmax_agreement(poly_plain, logits),
        "error_metrics": err,
        "operation_counts": {"matrix_multiply": 3, "add": 3, "polynomial": 1, "mask": 0},
        "polynomial_degree": args.degree,
        "block_count_first_layer": 2,
        "hidden_padding_ratio": 16 / 32,
        "output_padding_ratio": 22 / 32,
        "used_batches": used_batches,
        "used_columns_last_batch": layout["used_columns_last_batch"],
        **packing_stats(n, 64, shape),
        "level_log": level_log,
        "timing_log": timing,
        "runtime_total_s": runtime_total,
        "runtime_per_sample_amortized_s": runtime_total / n,
    }
    save_outputs("gl_blocked_digits64_packed_results", n, meta, rows)
    print(RESULTS / f"gl_blocked_digits64_packed_results_n{n}.json")


if __name__ == "__main__":
    main()
