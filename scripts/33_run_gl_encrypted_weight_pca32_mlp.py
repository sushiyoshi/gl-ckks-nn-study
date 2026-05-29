from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_encrypted_weight import encrypted_weight_mlp, failure_payload
from src.logging_utils import RESULTS, write_csv, write_json


def flatten(meta: dict) -> dict:
    timing = meta.get("timing_log") or {}
    return {
        "condition": "gl_encrypted_weight_pca32_mlp",
        "n_samples": meta.get("n_samples"),
        "ok": meta.get("ok"),
        "semantic_validation_passed": meta.get("semantic_validation_passed"),
        "task_type": meta.get("task_type", "mlp_poly_relu"),
        "weight_privacy": meta.get("weight_privacy", "encrypted_weight"),
        "activation": meta.get("activation"),
        "polynomial_degree": meta.get("polynomial_degree"),
        "ciphertext_count": meta.get("ciphertext_count"),
        "runtime_total_s": meta.get("runtime_total_s"),
        "runtime_per_sample_s": meta.get("runtime_per_sample_s"),
        "encryption_weight_s": timing.get("encryption_weight_s"),
        "encryption_input_s": timing.get("encryption_input_s"),
        "linear1_matrix_multiply_s": timing.get("linear1_matrix_multiply_s"),
        "activation_s": timing.get("activation_s"),
        "linear2_matrix_multiply_s": timing.get("linear2_matrix_multiply_s"),
        "decryption_s": timing.get("decryption_s"),
        "relative_l2": meta.get("logits_relative_l2"),
        "logits_linf": meta.get("logits_linf"),
        "logits_mae": meta.get("logits_mae"),
        "accuracy": meta.get("accuracy"),
        "argmax_agreement": meta.get("argmax_agreement"),
        "failure_category": meta.get("failure_category"),
        "failure_reason": meta.get("exception"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=1)
    args = parser.parse_args()
    try:
        meta = encrypted_weight_mlp(args.n_samples, args.degree)
    except Exception as exc:
        meta = failure_payload(
            exc,
            n_samples=args.n_samples,
            task_type="mlp_poly_relu",
            weight_privacy="encrypted_weight",
            activation=f"degree{args.degree}_poly",
            polynomial_degree=args.degree,
        )
    base = RESULTS / f"gl_encrypted_weight_pca32_mlp_n{args.n_samples}"
    write_json(base.with_suffix(".json"), meta)
    write_csv(base.with_suffix(".csv"), [flatten(meta)])
    print(base.with_suffix(".json"))


if __name__ == "__main__":
    main()
