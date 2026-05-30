from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_block_schedule import mlp_block_schedule
from src.gl_generic_mlp import (
    failure_payload,
    logical_poly_mlp,
    run_encrypted_weight_generic_mlp,
    run_precision_audit_generic_mlp,
    validate_plain_gl_layout,
)
from src.gl_shape_selector import (
    DEFAULT_ALLOWED_BATCHES,
    SUPPORTED_GL_SHAPES,
    choose_shape,
    choose_supported_shape,
    mlp_utilization_stats,
    normalize_shape,
)
from src.logging_utils import RESULTS, now_seconds, elapsed, write_csv, write_json
from src.precision_audit import write_precision_audit
from src.polynomial import fit_relu_power_polynomial


def make_synthetic(n_in: int, n_hidden: int, n_out: int, n_samples: int, seed: int, scale: float):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, scale, size=(n_samples, n_in)).astype(np.float64)
    w1 = rng.normal(0.0, scale / max(n_in, 1) ** 0.5, size=(n_in, n_hidden)).astype(np.float64)
    b1 = rng.normal(0.0, scale * 0.05, size=(n_hidden,)).astype(np.float64)
    w2 = rng.normal(0.0, scale / max(n_hidden, 1) ** 0.5, size=(n_hidden, n_out)).astype(np.float64)
    b2 = rng.normal(0.0, scale * 0.05, size=(n_out,)).astype(np.float64)
    return x, w1, b1, w2, b2


def flatten(meta: dict) -> dict:
    timing = meta.get("timing_log") or {}
    schedule = meta.get("schedule") or {}
    return {
        "condition": "gl_generic_synthetic_mlp",
        "ok": meta.get("ok"),
        "semantic_validation_passed": meta.get("semantic_validation_passed"),
        "failure_category": meta.get("failure_category"),
        "task_type": meta.get("task_type", "generic_mlp_poly_relu"),
        "dims": meta.get("dims"),
        "n_samples": meta.get("n_samples"),
        "block_size": meta.get("block_size"),
        "shape": meta.get("shape"),
        "selected_shape": meta.get("selected_shape") or meta.get("shape"),
        "input_blocks": schedule.get("input_blocks"),
        "hidden_blocks": schedule.get("hidden_blocks"),
        "output_blocks": schedule.get("output_blocks"),
        "block_matmul_count": meta.get("block_matmul_count") or schedule.get("total_linear_block_matmuls"),
        "activation_blocks": schedule.get("activation_blocks"),
        "sample_packing_utilization": meta.get("sample_packing_utilization"),
        "sample_capacity_per_pack": meta.get("sample_capacity_per_pack"),
        "input_entry_utilization_bounded": meta.get("input_entry_utilization_bounded"),
        "hidden_entry_utilization_bounded": meta.get("hidden_entry_utilization_bounded"),
        "output_entry_utilization_bounded": meta.get("output_entry_utilization_bounded"),
        "selected_shape_source": (meta.get("shape_selection") or {}).get("source"),
        "server_only_s": meta.get("server_only_s"),
        "total_s": meta.get("total_s"),
        "total_minus_keygen_s": meta.get("total_minus_keygen_s"),
        "server_per_sample_s": meta.get("server_per_sample_s"),
        "no_keygen_per_sample_s": meta.get("no_keygen_per_sample_s"),
        "relative_l2": meta.get("relative_l2"),
        "linf": meta.get("linf"),
        "mae": meta.get("mae"),
        "allclose": meta.get("allclose"),
        "encryption_input_s": timing.get("encryption_input_s"),
        "encryption_weight_s": timing.get("encryption_weight_s"),
        "activation_s": sum(v for k, v in timing.items() if k.startswith("activation_block")),
        "decryption_s": timing.get("decryption_s"),
        "failure_reason": meta.get("exception"),
    }


def dry_run_payload(args: argparse.Namespace, schedule: dict, radius: float, plain_ok: bool, total_s: float) -> dict:
    shape = tuple(schedule["shape"])
    return {
        "ok": True,
        "semantic_validation_passed": bool(plain_ok),
        "failure_category": None,
        "dry_run": True,
        "task_type": "generic_mlp_poly_relu",
        "dims": [args.n_in, args.n_hidden, args.n_out],
        "n_samples": args.n_samples,
        "block_size": args.block_size,
        "shape": list(shape),
        "schedule": schedule,
        "activation": "degree3_poly",
        "polynomial_degree": 3,
        "polynomial_radius": radius,
        "ciphertext_count": 0,
        "block_matmul_count": schedule["total_linear_block_matmuls"],
        **mlp_utilization_stats(args.n_samples, args.n_in, args.n_hidden, args.n_out, shape, args.block_size),
        "timing_log": {"dry_run_total_s": total_s},
        "server_only_s": 0.0,
        "total_s": total_s,
        "total_minus_keygen_s": total_s,
        "server_per_sample_s": 0.0,
        "no_keygen_per_sample_s": total_s / args.n_samples,
        "relative_l2": 0.0,
        "linf": 0.0,
        "mae": 0.0,
        "allclose": bool(plain_ok),
        "level_log": [],
    }


def parse_allowed_batches(text: str | None) -> list[int]:
    if not text:
        return list(DEFAULT_ALLOWED_BATCHES)
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_shape(text: str | None) -> tuple[int, int, int] | None:
    if not text:
        return None
    return normalize_shape(int(part.strip()) for part in text.split(",") if part.strip())


def selected_shape(args: argparse.Namespace) -> tuple[tuple[int, int, int], dict[str, Any]]:
    allowed = parse_allowed_batches(args.allowed_batches)
    explicit_shape = parse_shape(args.shape)
    if args.shape_policy == "exact":
        selection = choose_supported_shape(
            args.n_samples,
            args.n_in,
            args.n_hidden,
            args.n_out,
            policy="exact",
            explicit_shape=explicit_shape,
            probe_results_path=args.shape_probe_results,
        )
        selection["source"] = "shape_policy_exact"
        return tuple(selection["shape"]), selection
    if args.shape_policy in {"fixed_32", "min_block_matmuls", "balanced"}:
        selection = choose_supported_shape(
            args.n_samples,
            args.n_in,
            args.n_hidden,
            args.n_out,
            policy=args.shape_policy,
            probe_results_path=args.shape_probe_results,
        )
        selection["source"] = f"shape_policy_{args.shape_policy}"
        return tuple(selection["shape"]), selection
    if args.shape_batches is not None:
        shape = (int(args.shape_batches), int(args.block_size), int(args.block_size))
        return shape, {
            "source": "shape_batches",
            "shape": list(shape),
            "block_size": int(args.block_size),
            "allowed_batches": allowed,
        }
    if args.auto_shape:
        selection = choose_shape(
            args.n_samples,
            args.n_in,
            args.n_hidden,
            args.n_out,
            policy="balanced",
            probe_results_path=args.shape_probe_results,
        )
        selection["source"] = "auto_shape"
        selection["allowed_batches"] = allowed
        return tuple(selection["shape"]), selection
    shape = (256, int(args.block_size), int(args.block_size))
    return shape, {
        "source": "fixed_default",
        "shape": list(shape),
        "block_size": int(args.block_size),
        "allowed_batches": allowed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-in", type=int, required=True)
    parser.add_argument("--n-hidden", type=int, required=True)
    parser.add_argument("--n-out", type=int, required=True)
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-block-matmuls", type=int, default=128)
    parser.add_argument("--audit-precision", action="store_true")
    parser.add_argument("--auto-shape", action="store_true")
    parser.add_argument("--shape-policy", choices=["fixed_32", "min_block_matmuls", "balanced", "exact"])
    parser.add_argument("--shape")
    parser.add_argument("--shape-batches", type=int)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--allowed-batches", default=",".join(str(v) for v in DEFAULT_ALLOWED_BATCHES))
    parser.add_argument("--shape-probe-results", default=str(RESULTS / "glengine_shape_probe.json"))
    args = parser.parse_args()

    base = RESULTS / f"gl_generic_synthetic_{args.out_name}"
    audit_payload = None
    shape = (256, args.block_size, args.block_size)
    shape_selection: dict[str, Any] = {}
    try:
        t = now_seconds()
        shape, shape_selection = selected_shape(args)
        args.block_size = int(shape_selection.get("block_size") or shape[1])
        x, w1, b1, w2, b2 = make_synthetic(
            args.n_in, args.n_hidden, args.n_out, args.n_samples, args.seed, args.scale
        )
        z1 = x @ w1 + b1
        radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z1), 0.995)))))
        coeffs = fit_relu_power_polynomial(3, (-radius, radius)).astype(np.float64)
        schedule = mlp_block_schedule(args.n_in, args.n_hidden, args.n_out, args.n_samples, block_size=args.block_size, shape=shape)
        plain_validation = validate_plain_gl_layout(x, w1, b1, w2, b2, coeffs, shape=shape, block_size=args.block_size)
        if args.dry_run:
            meta = dry_run_payload(args, schedule, radius, plain_validation["ok"], elapsed(t))
        elif schedule["total_linear_block_matmuls"] > args.max_block_matmuls:
            raise RuntimeError(
                f"block matmul count {schedule['total_linear_block_matmuls']} exceeds max_block_matmuls={args.max_block_matmuls}; use --dry-run to write schedule only"
            )
        else:
            meta = run_encrypted_weight_generic_mlp(
                x,
                w1,
                b1,
                w2,
                b2,
                coeffs,
                radius,
                task_type="generic_mlp_poly_relu",
                metadata={
                    "synthetic": True,
                    "seed": args.seed,
                    "scale": args.scale,
                    "baseline": "semantic validation compares against the degree3_poly MLP baseline.",
                    "legacy_matrix_entry_utilization_input": "matrix_entry_utilization_input is retained for backward compatibility; use *_bounded metrics for 0..1 utilization.",
                },
                max_block_matmuls=args.max_block_matmuls,
                shape=shape,
                block_size=args.block_size,
            )
            if args.audit_precision:
                audit_payload = run_precision_audit_generic_mlp(
                    x,
                    w1,
                    b1,
                    w2,
                    b2,
                    coeffs,
                    radius,
                    task_type="generic_mlp_poly_relu",
                    metadata={
                        "synthetic": True,
                        "seed": args.seed,
                        "scale": args.scale,
                        "out_name": args.out_name,
                        "benchmark_result": str(base.with_suffix(".json")),
                    },
                    max_block_matmuls=args.max_block_matmuls,
                )
        meta["shape_selection"] = shape_selection
        meta["selected_shape"] = list(shape)
        meta["shape_probe_results"] = args.shape_probe_results
    except Exception as exc:
        utilization: dict[str, Any] = {}
        try:
            utilization = mlp_utilization_stats(args.n_samples, args.n_in, args.n_hidden, args.n_out, shape, args.block_size)
        except Exception:
            utilization = {}
        meta = failure_payload(
            exc,
            task_type="generic_mlp_poly_relu",
            dims=[args.n_in, args.n_hidden, args.n_out],
            n_samples=args.n_samples,
            block_size=args.block_size,
            shape=list(shape),
            selected_shape=list(shape),
            shape_selection=shape_selection,
            schedule=mlp_block_schedule(args.n_in, args.n_hidden, args.n_out, args.n_samples, block_size=args.block_size, shape=shape),
            activation="degree3_poly",
            polynomial_degree=3,
            **utilization,
        )
    write_json(base.with_suffix(".json"), meta)
    write_csv(base.with_suffix(".csv"), [flatten(meta)])
    if audit_payload is not None:
        write_precision_audit(RESULTS / f"precision_audit_{args.out_name}.json", audit_payload)
    print(base.with_suffix(".json"))


if __name__ == "__main__":
    main()
