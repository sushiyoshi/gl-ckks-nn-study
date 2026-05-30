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

from src.gl_packing import broadcast_bias_to_columns, broadcast_weight_to_batches, pack_columns, unpack_logits
from src.logging_utils import RESULTS, json_default, write_json
from src.metrics import accuracy, argmax_agreement, error_metrics
from src.model_pca32 import load_pca32_model
from src.polynomial import eval_power_polynomial, fit_relu_power_polynomial


N_SAMPLES = 8192
LOGICAL_DIMS = [32, 32, 10]
GL_SHAPE = (256, 32, 32)
ENCRYPTED_REFERENCE = {
    "server_only_s": 9.750431624994235,
    "total_s": 18.13020904199948,
    "key_generation_s": 6.289987874999497,
    "total_minus_keygen_s": 11.840221166999982,
}


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

    model, arrays = load_pca32_model(Path("data/pca32_mlp.joblib"))
    if args.n_samples != N_SAMPLES:
        raise ValueError(f"this report baseline is fixed to n_samples={N_SAMPLES}, got {args.n_samples}")
    if arrays["x_test_32"].shape[0] < args.n_samples:
        raise ValueError(f"x_test_32 has only {arrays['x_test_32'].shape[0]} rows")

    x = np.asarray(arrays["x_test_32"][: args.n_samples], dtype=np.float64)
    y = np.asarray(arrays["y_test"][: args.n_samples])

    z_train = model.relu_forward(arrays["x_train_32"])["z1"]
    radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z_train), 0.995)))))
    coeffs = fit_relu_power_polynomial(3, (-radius, radius)).astype(np.float64)

    def logical_mlp() -> np.ndarray:
        z1 = x @ model.w1 + model.b1
        h1 = eval_power_polynomial(z1, coeffs)
        return h1 @ model.w2 + model.b2

    input_tensor, layout = pack_columns(x, GL_SHAPE)
    used_batches = layout["used_batches"]
    w1_tensor = broadcast_weight_to_batches(model.w1.T, GL_SHAPE, used_batches, 32, 32)
    w2_tensor = broadcast_weight_to_batches(model.w2.T, GL_SHAPE, used_batches, 10, 32)
    b1_tensor = broadcast_bias_to_columns(model.b1, GL_SHAPE, layout, 32)
    b2_tensor = broadcast_bias_to_columns(model.b2, GL_SHAPE, layout, 10)

    def gl_layout_mlp() -> np.ndarray:
        hidden = np.matmul(w1_tensor, input_tensor) + b1_tensor
        activated = eval_power_polynomial(hidden, coeffs)
        logits_tensor = np.matmul(w2_tensor, activated) + b2_tensor
        return unpack_logits(logits_tensor, layout, 10)

    logical_times, logical_logits = benchmark(logical_mlp, warmup=args.warmup, repeat=args.repeat)
    gl_times, gl_logits = benchmark(gl_layout_mlp, warmup=args.warmup, repeat=args.repeat)

    logical_stats = summarize(logical_times)
    gl_stats = summarize(gl_times)
    metrics = error_metrics(logical_logits, gl_logits, "gl_layout_vs_logical")

    logical_median = logical_stats["median_s"]
    gl_median = gl_stats["median_s"]
    slowdown = {
        "encrypted_server_only_over_plaintext_logical_median": ENCRYPTED_REFERENCE["server_only_s"] / logical_median,
        "encrypted_no_keygen_over_plaintext_logical_median": ENCRYPTED_REFERENCE["total_minus_keygen_s"] / logical_median,
        "encrypted_server_only_over_plaintext_gl_layout_median": ENCRYPTED_REFERENCE["server_only_s"] / gl_median,
        "encrypted_no_keygen_over_plaintext_gl_layout_median": ENCRYPTED_REFERENCE["total_minus_keygen_s"] / gl_median,
    }

    payload: dict[str, Any] = {
        "n_samples": args.n_samples,
        "logical_dims": LOGICAL_DIMS,
        "gl_shape": list(GL_SHAPE),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "model_path": "data/pca32_mlp.joblib",
        "coefficient_source": "src.gl_encrypted_weight.encrypted_weight_mlp-compatible fit from pca32 train z1",
        "polynomial_degree": 3,
        "polynomial_radius": radius,
        "coefficients_power_basis_ascending": coeffs.tolist(),
        "used_batches": used_batches,
        "logical_mlp": logical_stats,
        "gl_layout_mlp": gl_stats,
        "encrypted_reference": ENCRYPTED_REFERENCE,
        "slowdown": slowdown,
        "validation": {
            **metrics,
            "allclose": bool(np.allclose(logical_logits, gl_logits, rtol=1e-12, atol=1e-12)),
            "logical_accuracy": accuracy(logical_logits, y),
            "argmax_agreement": argmax_agreement(logical_logits, gl_logits),
            "logical_checksum": float(np.sum(logical_logits)),
            "gl_layout_checksum": float(np.sum(gl_logits)),
        },
    }

    base = RESULTS / "plaintext_vs_gl_encrypted_mlp_n8192"
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
        **{f"encrypted_reference_{key}": value for key, value in ENCRYPTED_REFERENCE.items()},
        **slowdown,
        **payload["validation"],
    }
    write_csv(base.with_suffix(".csv"), json.loads(json.dumps(row, default=json_default)))
    print(base.with_suffix(".json"))


if __name__ == "__main__":
    main()
