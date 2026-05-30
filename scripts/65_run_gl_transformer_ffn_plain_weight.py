from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_block_schedule import mlp_block_schedule
from src.gl_generic_mlp import failure_payload, run_plain_weight_generic_mlp
from src.gl_shape_selector import choose_supported_shape, estimate_memory_for_shape, mlp_utilization_stats, normalize_shape
from src.logging_utils import RESULTS, elapsed, now_seconds, write_csv, write_json
from src.polynomial import fit_relu_power_polynomial


def make_synthetic(n_in: int, n_hidden: int, n_out: int, n_samples: int, seed: int, scale: float):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, scale, size=(n_samples, n_in)).astype(np.float64)
    w1 = rng.normal(0.0, scale / max(n_in, 1) ** 0.5, size=(n_in, n_hidden)).astype(np.float64)
    b1 = rng.normal(0.0, scale * 0.05, size=(n_hidden,)).astype(np.float64)
    w2 = rng.normal(0.0, scale / max(n_hidden, 1) ** 0.5, size=(n_hidden, n_out)).astype(np.float64)
    b2 = rng.normal(0.0, scale * 0.05, size=(n_out,)).astype(np.float64)
    return x, w1, b1, w2, b2


def parse_shape(text: str) -> tuple[int, int, int]:
    return normalize_shape(int(part.strip()) for part in text.split(",") if part.strip())


def flatten(meta: dict[str, Any]) -> dict[str, Any]:
    schedule = meta.get("schedule") or {}
    timing = meta.get("timing_log") or {}
    return {
        "ok": meta.get("ok"),
        "execution_mode": meta.get("execution_mode"),
        "failure_category": meta.get("failure_category"),
        "dims": meta.get("dims"),
        "n_samples": meta.get("n_samples"),
        "selected_shape": meta.get("selected_shape") or meta.get("shape"),
        "block_size": meta.get("block_size"),
        "sample_capacity": meta.get("sample_capacity") or meta.get("sample_capacity_per_pack"),
        "sample_packs": meta.get("sample_packs"),
        "sample_packing_utilization": meta.get("sample_packing_utilization"),
        "input_blocks": schedule.get("input_blocks"),
        "hidden_blocks": schedule.get("hidden_blocks"),
        "output_blocks": schedule.get("output_blocks"),
        "total_block_matmuls": meta.get("total_block_matmuls")
        or meta.get("block_matmul_count")
        or schedule.get("total_linear_block_matmuls"),
        "activation_blocks": schedule.get("activation_blocks"),
        "weight_privacy": meta.get("weight_privacy"),
        "server_only_s": meta.get("server_only_s"),
        "total_s": meta.get("total_s"),
        "total_minus_keygen_s": meta.get("total_minus_keygen_s"),
        "server_per_sample_s": meta.get("server_per_sample_s"),
        "relative_l2": meta.get("relative_l2"),
        "linf": meta.get("linf"),
        "mae": meta.get("mae"),
        "allclose": meta.get("allclose"),
        "encryption_input_s": timing.get("encryption_input_s"),
        "plaintext_weight_encode_s": timing.get("plaintext_weight_encode_s"),
        "decryption_s": timing.get("decryption_s"),
        "failure_reason": meta.get("exception"),
    }


def dry_run_payload(args: argparse.Namespace, shape: tuple[int, int, int], radius: float, total_s: float) -> dict[str, Any]:
    n_in, n_hidden, n_out = args.n_in, args.n_hidden, args.n_out
    block_size = shape[1]
    schedule = mlp_block_schedule(n_in, n_hidden, n_out, args.n_samples, block_size=block_size, shape=shape)
    memory = estimate_memory_for_shape(args.n_samples, n_in, n_hidden, n_out, shape=shape, block_size=block_size)
    return {
        "ok": True,
        "dry_run": True,
        "execution_mode": "dry_run",
        "semantic_validation_passed": None,
        "failure_category": None,
        "task_type": "transformer_ffn_component_poly_relu",
        "weight_privacy": "plaintext_weight",
        "dims": [n_in, n_hidden, n_out],
        "n_samples": args.n_samples,
        "shape": list(shape),
        "selected_shape": list(shape),
        "block_size": block_size,
        "sample_capacity": shape[0] * shape[2],
        "schedule": schedule,
        "input_blocks": schedule["input_blocks"],
        "hidden_blocks": schedule["hidden_blocks"],
        "output_blocks": schedule["output_blocks"],
        "total_block_matmuls": schedule["total_linear_block_matmuls"],
        "activation_blocks": schedule["activation_blocks"],
        "block_matmul_count": schedule["total_linear_block_matmuls"],
        "activation": "degree3_poly",
        "polynomial_degree": 3,
        "polynomial_radius": radius,
        "peak_ciphertext_count_estimate": memory["peak_ciphertext_count_estimate"],
        **mlp_utilization_stats(args.n_samples, n_in, n_hidden, n_out, shape, block_size),
        "level_log": [],
        "timing_log": {"dry_run_total_s": total_s},
        "server_only_s": 0.0,
        "total_s": total_s,
        "total_minus_keygen_s": total_s,
        "relative_l2": None,
        "linf": None,
        "mae": None,
        "allclose": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-in", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=2048)
    parser.add_argument("--n-out", type=int, default=512)
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--shape", default="16,512,512")
    parser.add_argument("--shape-policy", choices=["exact", "min_block_matmuls", "balanced", "fixed_32"], default="exact")
    parser.add_argument("--max-block-matmuls", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shape-probe-results", default=str(RESULTS / "gl_large_shape_probe.json"))
    args = parser.parse_args()

    t = now_seconds()
    requested_shape = parse_shape(args.shape)
    shape_selection: dict[str, Any] = {}
    shape = requested_shape
    suffix = "plain_weight_dryrun" if args.dry_run else "plain_weight_run"
    base = RESULTS / f"gl_transformer_ffn_{args.n_in}_{args.n_hidden}_{args.n_out}_shape_{shape[0]}_{shape[1]}_{shape[2]}_{suffix}"
    try:
        shape_selection = choose_supported_shape(
            args.n_samples,
            args.n_in,
            args.n_hidden,
            args.n_out,
            policy=args.shape_policy,
            explicit_shape=requested_shape if args.shape_policy == "exact" else None,
            probe_results_path=args.shape_probe_results,
        )
        shape = tuple(shape_selection["shape"])
        block_size = int(shape_selection["block_size"])
        if args.dry_run:
            radius = 3.0
            meta = dry_run_payload(args, shape, radius, elapsed(t))
        else:
            x, w1, b1, w2, b2 = make_synthetic(args.n_in, args.n_hidden, args.n_out, args.n_samples, args.seed, args.scale)
            z1 = x @ w1 + b1
            radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z1), 0.995)))))
            coeffs = fit_relu_power_polynomial(3, (-radius, radius)).astype(np.float64)
            meta = run_plain_weight_generic_mlp(
                x,
                w1,
                b1,
                w2,
                b2,
                coeffs,
                radius,
                task_type="transformer_ffn_component_poly_relu",
                metadata={
                    "synthetic": True,
                    "seed": args.seed,
                    "scale": args.scale,
                    "component": "Transformer FFN only; no attention, softmax, layernorm, or bootstrapping.",
                },
                max_block_matmuls=args.max_block_matmuls,
                shape=shape,
                block_size=block_size,
            )
            meta["execution_mode"] = "actual"
        meta["shape_selection"] = shape_selection
        meta["selected_shape"] = list(shape)
        meta["sample_capacity"] = shape[0] * shape[2]
        meta["total_block_matmuls"] = (meta.get("schedule") or {}).get("total_linear_block_matmuls")
        meta["peak_ciphertext_count_estimate"] = estimate_memory_for_shape(
            args.n_samples, args.n_in, args.n_hidden, args.n_out, shape=shape, block_size=shape[1]
        )["peak_ciphertext_count_estimate"]
    except BaseException as exc:
        utilization: dict[str, Any] = {}
        try:
            utilization = mlp_utilization_stats(args.n_samples, args.n_in, args.n_hidden, args.n_out, shape, shape[1])
        except Exception:
            utilization = {}
        schedule = mlp_block_schedule(args.n_in, args.n_hidden, args.n_out, args.n_samples, block_size=shape[1], shape=shape)
        meta = failure_payload(
            exc,
            execution_mode="dry_run" if args.dry_run else "actual",
            task_type="transformer_ffn_component_poly_relu",
            weight_privacy="plaintext_weight",
            dims=[args.n_in, args.n_hidden, args.n_out],
            n_samples=args.n_samples,
            shape=list(shape),
            selected_shape=list(shape),
            block_size=shape[1],
            sample_capacity=shape[0] * shape[2],
            schedule=schedule,
            shape_selection=shape_selection,
            total_block_matmuls=schedule["total_linear_block_matmuls"],
            activation="degree3_poly",
            polynomial_degree=3,
            traceback=traceback.format_exc(limit=8),
            **utilization,
        )
    write_json(base.with_suffix(".json"), meta)
    write_csv(base.with_suffix(".csv"), [flatten(meta)])
    print(base.with_suffix(".json"))


if __name__ == "__main__":
    main()
