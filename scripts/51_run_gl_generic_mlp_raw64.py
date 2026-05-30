from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_block_mlp import raw64_polynomial
from src.gl_generic_mlp import failure_payload, run_encrypted_weight_generic_mlp, run_precision_audit_generic_mlp
from src.logging_utils import RESULTS, read_json, write_csv, write_json
from src.metrics import argmax_agreement
from src.model_pca32 import require_sample_count
from src.model_raw64 import load_raw64_model
from src.precision_audit import write_precision_audit


def flatten(meta: dict) -> dict:
    timing = meta.get("timing_log") or {}
    schedule = meta.get("schedule") or {}
    return {
        "condition": "gl_generic_raw64_mlp",
        "n_samples": meta.get("n_samples"),
        "ok": meta.get("ok"),
        "semantic_validation_passed": meta.get("semantic_validation_passed"),
        "failure_category": meta.get("failure_category"),
        "task_type": meta.get("task_type", "generic_mlp_poly_relu"),
        "weight_privacy": meta.get("weight_privacy", "encrypted_weight"),
        "activation": meta.get("activation", "degree3_poly"),
        "polynomial_degree": meta.get("polynomial_degree"),
        "polynomial_radius": meta.get("polynomial_radius"),
        "logical_dims": meta.get("logical_dims"),
        "shape": meta.get("shape"),
        "input_blocks": schedule.get("input_blocks"),
        "hidden_blocks": schedule.get("hidden_blocks"),
        "output_blocks": schedule.get("output_blocks"),
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
        "activation_s": sum(v for k, v in timing.items() if k.startswith("activation_block")),
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
        "raw64_reference_relative_l2": (meta.get("raw64_reference_comparison") or {}).get("relative_l2_delta"),
        "raw64_reference_argmax_agreement_delta": (meta.get("raw64_reference_comparison") or {}).get("argmax_agreement_delta"),
        "raw64_reference_server_only_s_delta": (meta.get("raw64_reference_comparison") or {}).get("server_only_s_delta"),
        "raw64_reference_schedule_matches": (meta.get("raw64_reference_comparison") or {}).get("schedule_matches"),
        "failure_reason": meta.get("exception"),
    }


def attach_reference_comparison(meta: dict, n_samples: int) -> None:
    ref_path = RESULTS / f"gl_encrypted_weight_raw64_mlp_n{n_samples}.json"
    if not ref_path.exists() or not meta.get("ok"):
        meta["raw64_reference_comparison"] = {"available": False, "path": str(ref_path)}
        return
    ref = read_json(ref_path)
    schedule = meta.get("schedule") or {}
    meta["raw64_reference_comparison"] = {
        "available": True,
        "path": str(ref_path),
        "relative_l2_delta": None
        if ref.get("logits_relative_l2") is None
        else meta.get("logits_relative_l2") - ref.get("logits_relative_l2"),
        "argmax_agreement_delta": None
        if ref.get("argmax_agreement") is None
        else meta.get("argmax_agreement") - ref.get("argmax_agreement"),
        "server_only_s_delta": None
        if ref.get("server_only_s") is None
        else meta.get("server_only_s") - ref.get("server_only_s"),
        "schedule_matches": bool(
            schedule.get("input_blocks") == 2
            and schedule.get("hidden_blocks") == 1
            and schedule.get("output_blocks") == 1
            and schedule.get("total_linear_block_matmuls") == ref.get("block_matmul_count")
        ),
        "reference_server_only_s": ref.get("server_only_s"),
        "generic_server_only_s": meta.get("server_only_s"),
        "reference_relative_l2": ref.get("logits_relative_l2"),
        "generic_relative_l2": meta.get("logits_relative_l2"),
        "reference_argmax_agreement": ref.get("argmax_agreement"),
        "generic_argmax_agreement": meta.get("argmax_agreement"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--audit-precision", action="store_true")
    args = parser.parse_args()
    audit_payload = None
    try:
        model, arrays = load_raw64_model()
        require_sample_count(args.n_samples, arrays["x_test_64"].shape[0], label="x_test_64")
        radius, coeffs = raw64_polynomial()
        meta = run_encrypted_weight_generic_mlp(
            arrays["x_test_64"][: args.n_samples],
            model.w1,
            model.b1,
            model.w2,
            model.b2,
            coeffs,
            radius,
            y=arrays["y_test"][: args.n_samples],
            task_type="generic_raw64_mlp_poly_relu",
            metadata={
                "baseline": "semantic validation compares against the degree3_poly MLP baseline, not the original ReLU MLP.",
                "model_path": "data/raw64_mlp.joblib",
                "reference_script": "scripts/48_run_gl_encrypted_weight_raw64_mlp.py",
            },
        )
        attach_reference_comparison(meta, args.n_samples)
        if args.audit_precision:
            audit_payload = run_precision_audit_generic_mlp(
                arrays["x_test_64"][: args.n_samples],
                model.w1,
                model.b1,
                model.w2,
                model.b2,
                coeffs,
                radius,
                task_type="generic_raw64_mlp_poly_relu",
                metadata={
                    "baseline": "semantic validation compares against the degree3_poly MLP baseline, not the original ReLU MLP.",
                    "model_path": "data/raw64_mlp.joblib",
                    "benchmark_result": str(RESULTS / f"gl_generic_raw64_mlp_n{args.n_samples}.json"),
                },
            )
    except Exception as exc:
        meta = failure_payload(
            exc,
            n_samples=args.n_samples,
            task_type="generic_raw64_mlp_poly_relu",
            weight_privacy="encrypted_weight",
            activation="degree3_poly",
            polynomial_degree=3,
            logical_dims=[64, 32, 10],
            dims=[64, 32, 10],
            shape=[256, 32, 32],
        )
    base = RESULTS / f"gl_generic_raw64_mlp_n{args.n_samples}"
    write_json(base.with_suffix(".json"), meta)
    write_csv(base.with_suffix(".csv"), [flatten(meta)])
    if audit_payload is not None:
        write_precision_audit(RESULTS / f"precision_audit_raw64_n{args.n_samples}.json", audit_payload)
    print(base.with_suffix(".json"))


if __name__ == "__main__":
    main()
