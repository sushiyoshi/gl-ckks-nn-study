from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ckks_packing import (
    make_block_diagonal_linear_matrix,
    make_repeated_bias_vector,
    pack_samples_blocks,
    packing_stats,
    unpack_logits_blocks,
)
from src.logging_utils import RESULTS, ct_info, elapsed, exception_record, now_seconds, write_csv, write_json
from src.metrics import accuracy, argmax_agreement, error_metrics
from src.model_pca32 import load_pca32_model, require_sample_count
from src.polynomial import fit_relu_power_polynomial


def write_unsupported(reason: str, details: dict) -> None:
    path = RESULTS / "ckks_pca32_packed_unsupported.md"
    path.write_text(f"# CKKS PCA32 packed unsupported\n\nReason: {reason}\n\nDetails:\n\n```json\n{details}\n```\n")


def save_outputs(meta: dict, rows: list[dict], n_samples: int) -> None:
    write_json(RESULTS / "ckks_pca32_packed_results.json", meta)
    write_csv(RESULTS / "ckks_pca32_packed_results.csv", rows)
    write_json(RESULTS / f"ckks_pca32_packed_results_n{n_samples}.json", meta)
    write_csv(RESULTS / f"ckks_pca32_packed_results_n{n_samples}.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=450)
    parser.add_argument("--chunk-samples", type=int)
    args = parser.parse_args()

    from desilofhe import Engine

    t_total = now_seconds()
    rows: list[dict] = []
    try:
        model, arrays = load_pca32_model()
        require_sample_count(args.n_samples, arrays["x_test_32"].shape[0], label="x_test_32")
        n = args.n_samples
        x = arrays["x_test_32"][:n]
        y = arrays["y_test"][:n]
        z_train = model.relu_forward(arrays["x_train_32"])["z1"]
        radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z_train), 0.995)))))
        coeffs = fit_relu_power_polynomial(args.degree, (-radius, radius)).astype(np.float64)
        poly_plain = model.poly_forward(x, coeffs)["logits"]

        engine = Engine()
        slot_count = int(engine.slot_count)
        block_dim = 32
        samples_per_ct_default = slot_count // block_dim
        chunk_samples = args.chunk_samples or samples_per_ct_default
        chunks, layout = pack_samples_blocks(x, block_dim, slot_count, chunk_samples)

        timing = {"key_generation_s": 0.0, "encryption_s_total": 0.0, "linear1_s_total": 0.0, "activation_s_total": 0.0, "linear2_s_total": 0.0, "decryption_s_total": 0.0}
        t = now_seconds()
        sk = engine.create_secret_key()
        rot_key = engine.create_rotation_key(sk)
        relin_key = engine.create_relinearization_key(sk)
        timing["key_generation_s"] = elapsed(t)

        samples_per_ct = layout["samples_per_ct"]
        t = now_seconds()
        m1 = make_block_diagonal_linear_matrix(model.w1.T, block_dim, samples_per_ct, slot_count, 32, 32)
        m2 = make_block_diagonal_linear_matrix(model.w2.T, block_dim, samples_per_ct, slot_count, 10, 32)
        b1 = make_repeated_bias_vector(model.b1, block_dim, samples_per_ct, slot_count, 32)
        b2 = make_repeated_bias_vector(model.b2, block_dim, samples_per_ct, slot_count, 10)
        matrix_build_s = elapsed(t)

        decoded_chunks = []
        level_log = []
        for chunk_index, chunk in enumerate(chunks):
            t = now_seconds()
            ct = engine.encrypt(chunk, sk)
            timing["encryption_s_total"] += elapsed(t)
            if chunk_index == 0:
                level_log.append({"stage": "input", **ct_info(ct)})
            t = now_seconds()
            ct = engine.multiply_matrix(m1, ct, rot_key)
            ct = engine.add(ct, b1)
            timing["linear1_s_total"] += elapsed(t)
            if chunk_index == 0:
                level_log.append({"stage": "linear1", **ct_info(ct)})
            t = now_seconds()
            ct = engine.evaluate_polynomial(ct, [float(c) for c in coeffs], relin_key)
            timing["activation_s_total"] += elapsed(t)
            if chunk_index == 0:
                level_log.append({"stage": "activation", **ct_info(ct)})
            t = now_seconds()
            ct = engine.multiply_matrix(m2, ct, rot_key)
            ct = engine.add(ct, b2)
            timing["linear2_s_total"] += elapsed(t)
            if chunk_index == 0:
                level_log.append({"stage": "linear2", **ct_info(ct)})
            t = now_seconds()
            decoded_chunks.append(np.asarray(engine.decrypt(ct, sk), dtype=np.float64))
            timing["decryption_s_total"] += elapsed(t)

        logits = unpack_logits_blocks(decoded_chunks, layout, 10)
        err = error_metrics(poly_plain, logits, "logits")
        runtime_total = elapsed(t_total)
        rows = [
            {
                "sample_index": i,
                "true_label": int(y[i]),
                "poly_pred": int(np.argmax(poly_plain[i])),
                "ckks_pred": int(np.argmax(logits[i])),
            }
            for i in range(n)
        ]
        meta = {
            "ok": True,
            "semantic_validation_passed": bool(err["logits_relative_l2"] < 1e-5),
            "accuracy": accuracy(logits, y),
            "accuracy_poly_plain": accuracy(poly_plain, y),
            "argmax_agreement_poly_vs_fhe": argmax_agreement(poly_plain, logits),
            "error_metrics": err,
            "n_samples": n,
            "slot_count": slot_count,
            **packing_stats(n, block_dim, slot_count, len(chunks)),
            "output_padding_ratio": 22 / 32,
            "matrix_multiply_count": 2 * len(chunks),
            "add_count": 2 * len(chunks),
            "polynomial_degree": args.degree,
            "operation_counts": {"matrix_multiply": 2 * len(chunks), "add": 2 * len(chunks), "polynomial": len(chunks)},
            "matrix_build_s": matrix_build_s,
            "level_log": level_log,
            "timing_log": timing,
            "runtime_total_s": runtime_total,
            "runtime_per_sample_amortized_s": runtime_total / n,
            "memory_notes": "Dense numpy 8192x8192 block diagonal matrices are used for semantic validation.",
        }
        save_outputs(meta, rows, n)
        print(RESULTS / f"ckks_pca32_packed_results_n{n}.json")
    except Exception as exc:
        rec = exception_record(exc, api="CKKS PCA32 packed")
        rec["runtime_total_s"] = elapsed(t_total)
        write_unsupported("CKKS PCA32 packed path failed", rec)
        write_json(RESULTS / "ckks_pca32_packed_results.json", rec)
        write_csv(RESULTS / "ckks_pca32_packed_results.csv", rows)
        print(RESULTS / "ckks_pca32_packed_unsupported.md")


if __name__ == "__main__":
    main()
