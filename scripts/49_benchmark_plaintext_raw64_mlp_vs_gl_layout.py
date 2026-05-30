from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_block_mlp import block_structure, raw64_polynomial
from src.gl_packing import (
    broadcast_bias_to_columns,
    broadcast_weight_to_batches,
    split64_to_two_packed_tensors,
    unpack_logits,
)
from src.logging_utils import RESULTS, json_default, read_json, write_json
from src.metrics import accuracy, argmax_agreement, error_metrics
from src.model_raw64 import load_raw64_model
from src.polynomial import eval_power_polynomial


N_SAMPLES = 8192
LOGICAL_DIMS = [64, 32, 10]
GL_SHAPE = (256, 32, 32)


def summarize(times: list[float]) -> dict[str, float]:
    return {
        "min_s": min(times),
        "median_s": statistics.median(times),
        "mean_s": statistics.fmean(times),
        "max_s": max(times),
    }


def benchmark(fn: Callable[[], np.ndarray], *, warmup: int, repeat: int) -> tuple[list[float], np.ndarray]:
    out: np.ndarray | None = None
    for _ in range(warmup):
        out = fn()
    times: list[float] = []
    for _ in range(repeat):
        start = perf_counter()
        out = fn()
        times.append(perf_counter() - start)
    if out is None:
        out = fn()
    return times, out


def flatten(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()
    if args.n_samples != N_SAMPLES:
        raise ValueError(f"this report baseline is fixed to n_samples={N_SAMPLES}, got {args.n_samples}")

    model, arrays = load_raw64_model(Path("data/raw64_mlp.joblib"))
    if arrays["x_test_64"].shape[0] < args.n_samples:
        raise ValueError(f"x_test_64 has only {arrays['x_test_64'].shape[0]} rows")
    x = np.asarray(arrays["x_test_64"][: args.n_samples], dtype=np.float64)
    y = np.asarray(arrays["y_test"][: args.n_samples])
    radius, coeffs = raw64_polynomial()

    def logical_mlp() -> np.ndarray:
        z1 = x @ model.w1 + model.b1
        h1 = eval_power_polynomial(z1, coeffs)
        return h1 @ model.w2 + model.b2

    x0_tensor, x1_tensor, layout = split64_to_two_packed_tensors(x, GL_SHAPE)
    used_batches = layout["used_batches"]
    w1_t = model.w1.T
    w1a_tensor = broadcast_weight_to_batches(w1_t[:, :32], GL_SHAPE, used_batches, 32, 32)
    w1b_tensor = broadcast_weight_to_batches(w1_t[:, 32:64], GL_SHAPE, used_batches, 32, 32)
    w2_tensor = broadcast_weight_to_batches(model.w2.T, GL_SHAPE, used_batches, 10, 32)
    b1_tensor = broadcast_bias_to_columns(model.b1, GL_SHAPE, layout, 32)
    b2_tensor = broadcast_bias_to_columns(model.b2, GL_SHAPE, layout, 10)

    def gl_layout_mlp() -> np.ndarray:
        hidden = np.matmul(w1a_tensor, x0_tensor) + np.matmul(w1b_tensor, x1_tensor) + b1_tensor
        activated = eval_power_polynomial(hidden, coeffs)
        logits_tensor = np.matmul(w2_tensor, activated) + b2_tensor
        return unpack_logits(logits_tensor, layout, 10)

    logical_times, logical_logits = benchmark(logical_mlp, warmup=args.warmup, repeat=args.repeat)
    gl_times, gl_logits = benchmark(gl_layout_mlp, warmup=args.warmup, repeat=args.repeat)
    logical_stats = summarize(logical_times)
    gl_stats = summarize(gl_times)
    metrics = error_metrics(logical_logits, gl_logits, "gl_layout_vs_logical")

    encrypted_path = RESULTS / "gl_encrypted_weight_raw64_mlp_n8192.json"
    encrypted_reference = read_json(encrypted_path) if encrypted_path.exists() else {"ok": False, "missing": str(encrypted_path)}
    logical_median = logical_stats["median_s"]
    gl_median = gl_stats["median_s"]
    server_only = encrypted_reference.get("server_only_s")
    total_minus_keygen = encrypted_reference.get("total_minus_keygen_s")
    slowdown = {
        "encrypted_server_only_over_plaintext_logical_median": (server_only / logical_median) if server_only else None,
        "encrypted_no_keygen_over_plaintext_logical_median": (total_minus_keygen / logical_median) if total_minus_keygen else None,
        "encrypted_server_only_over_plaintext_gl_layout_median": (server_only / gl_median) if server_only else None,
        "encrypted_no_keygen_over_plaintext_gl_layout_median": (total_minus_keygen / gl_median) if total_minus_keygen else None,
    }

    payload: dict[str, Any] = {
        "n_samples": args.n_samples,
        "task_type": "raw64_mlp_poly_relu",
        "logical_dims": LOGICAL_DIMS,
        "gl_shape": list(GL_SHAPE),
        "block_structure": block_structure(),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "model_path": "data/raw64_mlp.joblib",
        "coefficient_source": "raw64 train pre-activation q0.995 radius fit",
        "polynomial_degree": 3,
        "polynomial_radius": radius,
        "coefficients": coeffs.tolist(),
        "used_batches": used_batches,
        "used_columns_last_batch": layout["used_columns_last_batch"],
        "logical_mlp": logical_stats,
        "gl_layout_mlp": gl_stats,
        "encrypted_reference": encrypted_reference,
        "slowdown": slowdown,
        "metadata": {
            "evaluation_sampling": "x_test_64/y_test are deterministic resamples of the base sklearn digits test split for throughput measurement.",
            "plaintext_gl_layout": "packs X0/X1 and broadcasts padded W1a/W1b/W2 into [256,32,32] tensors, then unpacks logits.",
        },
        "validation": {
            **metrics,
            "allclose": bool(np.allclose(logical_logits, gl_logits, rtol=1e-12, atol=1e-12)),
            "logical_accuracy": accuracy(logical_logits, y),
            "gl_layout_accuracy": accuracy(gl_logits, y),
            "argmax_agreement": argmax_agreement(logical_logits, gl_logits),
            "logical_checksum": float(np.sum(logical_logits)),
            "gl_layout_checksum": float(np.sum(gl_logits)),
        },
    }

    base = RESULTS / "plaintext_vs_gl_encrypted_raw64_mlp_n8192"
    write_json(base.with_suffix(".json"), payload)
    row = {
        "n_samples": payload["n_samples"],
        "logical_dims": json.dumps(LOGICAL_DIMS),
        "gl_shape": json.dumps(list(GL_SHAPE)),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "polynomial_degree": payload["polynomial_degree"],
        "polynomial_radius": payload["polynomial_radius"],
        **flatten("logical_mlp", logical_stats),
        **flatten("gl_layout_mlp", gl_stats),
        "encrypted_reference_ok": encrypted_reference.get("ok"),
        "encrypted_reference_server_only_s": encrypted_reference.get("server_only_s"),
        "encrypted_reference_total_minus_keygen_s": encrypted_reference.get("total_minus_keygen_s"),
        **slowdown,
        **payload["validation"],
    }
    write_csv(base.with_suffix(".csv"), json.loads(json.dumps(row, default=json_default)))
    print(base.with_suffix(".json"))


if __name__ == "__main__":
    main()
