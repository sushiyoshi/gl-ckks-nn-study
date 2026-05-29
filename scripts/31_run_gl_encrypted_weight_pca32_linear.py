from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_encrypted_weight import encrypted_weight_linear, failure_payload
from src.logging_utils import RESULTS, write_csv, write_json


def flatten(meta: dict) -> dict:
    timing = meta.get("timing_log") or {}
    return {
        "condition": "gl_encrypted_weight_pca32_linear",
        "n_samples": meta.get("n_samples"),
        "ok": meta.get("ok"),
        "semantic_validation_passed": meta.get("semantic_validation_passed"),
        "task_type": meta.get("task_type", "linear"),
        "weight_privacy": meta.get("weight_privacy", "encrypted_weight"),
        "activation": meta.get("activation", "none"),
        "ciphertext_count": meta.get("ciphertext_count"),
        "runtime_total_s": meta.get("runtime_total_s"),
        "runtime_per_sample_s": meta.get("runtime_per_sample_s"),
        "encryption_weight_s": timing.get("encryption_weight_s"),
        "encryption_input_s": timing.get("encryption_input_s"),
        "matrix_multiply_s": timing.get("matrix_multiply_s"),
        "decryption_s": timing.get("decryption_s"),
        "hidden_relative_l2": meta.get("hidden_relative_l2"),
        "hidden_linf": meta.get("hidden_linf"),
        "hidden_mae": meta.get("hidden_mae"),
        "failure_category": meta.get("failure_category"),
        "failure_reason": meta.get("exception"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1)
    args = parser.parse_args()
    try:
        meta = encrypted_weight_linear(args.n_samples)
    except Exception as exc:
        meta = failure_payload(exc, n_samples=args.n_samples, task_type="linear", weight_privacy="encrypted_weight", activation="none")
    base = RESULTS / f"gl_encrypted_weight_pca32_linear_n{args.n_samples}"
    write_json(base.with_suffix(".json"), meta)
    write_csv(base.with_suffix(".csv"), [flatten(meta)])
    print(base.with_suffix(".json"))


if __name__ == "__main__":
    main()
