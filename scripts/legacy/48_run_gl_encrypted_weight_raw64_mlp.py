from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_block_mlp import block_structure, encrypted_weight_raw64_mlp, failure_payload
from src.logging_utils import RESULTS, write_csv, write_json


def flatten(meta: dict) -> dict:
    timing = meta.get("timing_log") or {}
    return {
        "condition": "gl_encrypted_weight_raw64_mlp",
        "n_samples": meta.get("n_samples"),
        "ok": meta.get("ok"),
        "semantic_validation_passed": meta.get("semantic_validation_passed"),
        "failure_category": meta.get("failure_category"),
        "task_type": meta.get("task_type", "raw64_mlp_poly_relu"),
        "weight_privacy": meta.get("weight_privacy", "encrypted_weight"),
        "activation": meta.get("activation", "degree3_poly"),
        "polynomial_degree": meta.get("polynomial_degree"),
        "polynomial_radius": meta.get("polynomial_radius"),
        "logical_dims": meta.get("logical_dims"),
        "shape": meta.get("shape"),
        "ciphertext_count": meta.get("ciphertext_count"),
        "input_ciphertext_count": meta.get("input_ciphertext_count"),
        "weight_ciphertext_count": meta.get("weight_ciphertext_count"),
        "block_matmul_count": meta.get("block_matmul_count"),
        "sample_packing_utilization": meta.get("sample_packing_utilization"),
        "used_batches": meta.get("used_batches"),
        "used_columns_last_batch": meta.get("used_columns_last_batch"),
        "feature_dim": meta.get("feature_dim"),
        "samples_per_ciphertext_capacity": meta.get("samples_per_ciphertext_capacity"),
        "encryption_input_s": timing.get("encryption_input_s"),
        "encryption_weight_s": timing.get("encryption_weight_s"),
        "linear1_block0_matrix_multiply_s": timing.get("linear1_block0_matrix_multiply_s"),
        "linear1_block1_matrix_multiply_s": timing.get("linear1_block1_matrix_multiply_s"),
        "linear1_accumulation_s": timing.get("linear1_accumulation_s"),
        "activation_s": timing.get("activation_s"),
        "linear2_matrix_multiply_s": timing.get("linear2_matrix_multiply_s"),
        "decryption_s": timing.get("decryption_s"),
        "server_only_s": meta.get("server_only_s"),
        "total_s": meta.get("total_s"),
        "runtime_per_sample_s": meta.get("runtime_per_sample_s"),
        "total_minus_keygen_s": meta.get("total_minus_keygen_s"),
        "server_per_sample_s": meta.get("server_per_sample_s"),
        "no_keygen_per_sample_s": meta.get("no_keygen_per_sample_s"),
        "logits_relative_l2": meta.get("logits_relative_l2"),
        "logits_linf": meta.get("logits_linf"),
        "logits_mae": meta.get("logits_mae"),
        "logits_allclose": meta.get("logits_allclose"),
        "accuracy": meta.get("accuracy"),
        "argmax_agreement": meta.get("argmax_agreement"),
        "failure_reason": meta.get("exception"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1)
    args = parser.parse_args()
    try:
        meta = encrypted_weight_raw64_mlp(args.n_samples)
    except Exception as exc:
        meta = failure_payload(
            exc,
            n_samples=args.n_samples,
            task_type="raw64_mlp_poly_relu",
            weight_privacy="encrypted_weight",
            activation="degree3_poly",
            polynomial_degree=3,
            logical_dims=[64, 32, 10],
            shape=[256, 32, 32],
            block_matmul_count=3,
            block_structure=block_structure(),
        )
    base = RESULTS / f"gl_encrypted_weight_raw64_mlp_n{args.n_samples}"
    write_json(base.with_suffix(".json"), meta)
    write_csv(base.with_suffix(".csv"), [flatten(meta)])
    print(base.with_suffix(".json"))


if __name__ == "__main__":
    main()
