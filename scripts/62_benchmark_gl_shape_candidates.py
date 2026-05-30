from __future__ import annotations

import traceback
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_generic_mlp import failure_payload, run_encrypted_weight_generic_mlp, validate_plain_gl_layout
from src.gl_shape_selector import choose_shape, estimate_memory_for_shape
from src.logging_utils import RESULTS, elapsed, now_seconds, write_csv, write_json
from src.polynomial import fit_relu_power_polynomial


N_SAMPLES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DIMS = (96, 64, 10)
BLOCK_SIZE = 32
FIXED_SHAPE = (256, 32, 32)


def make_synthetic(n_in: int, n_hidden: int, n_out: int, n_samples: int, seed: int, scale: float):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, scale, size=(n_samples, n_in)).astype(np.float64)
    w1 = rng.normal(0.0, scale / max(n_in, 1) ** 0.5, size=(n_in, n_hidden)).astype(np.float64)
    b1 = rng.normal(0.0, scale * 0.05, size=(n_hidden,)).astype(np.float64)
    w2 = rng.normal(0.0, scale / max(n_hidden, 1) ** 0.5, size=(n_hidden, n_out)).astype(np.float64)
    b2 = rng.normal(0.0, scale * 0.05, size=(n_out,)).astype(np.float64)
    return x, w1, b1, w2, b2


def run_one(n_samples: int, mode: str, shape: tuple[int, int, int]) -> dict[str, Any]:
    n_in, n_hidden, n_out = DIMS
    t = now_seconds()
    try:
        x, w1, b1, w2, b2 = make_synthetic(n_in, n_hidden, n_out, n_samples, seed=42, scale=0.5)
        z1 = x @ w1 + b1
        radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(z1), 0.995)))))
        coeffs = fit_relu_power_polynomial(3, (-radius, radius)).astype(np.float64)
        plain = validate_plain_gl_layout(x, w1, b1, w2, b2, coeffs, shape=shape, block_size=BLOCK_SIZE)
        if not plain["ok"]:
            raise RuntimeError("plain GL-layout validation failed")
        meta = run_encrypted_weight_generic_mlp(
            x,
            w1,
            b1,
            w2,
            b2,
            coeffs,
            radius,
            metadata={"benchmark": "gl_shape_candidates", "mode": mode},
            max_block_matmuls=128,
            shape=shape,
            block_size=BLOCK_SIZE,
        )
        meta["mode"] = mode
        meta["selected_shape"] = list(shape)
        meta["peak_ciphertext_count_estimate"] = estimate_memory_for_shape(
            n_samples, n_in, n_hidden, n_out, shape=shape, block_size=BLOCK_SIZE
        )["peak_ciphertext_count_estimate"]
        return meta
    except BaseException as exc:
        payload = failure_payload(
            exc,
            mode=mode,
            dims=list(DIMS),
            n_samples=n_samples,
            selected_shape=list(shape),
            block_size=BLOCK_SIZE,
        )
        payload["traceback"] = traceback.format_exc(limit=6)
        payload["total_s"] = elapsed(t)
        return payload


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": row.get("mode"),
        "ok": row.get("ok"),
        "n_samples": row.get("n_samples"),
        "selected_shape": row.get("selected_shape") or row.get("shape"),
        "keygen_s": row.get("key_generation_s"),
        "encryption_s": (row.get("timing_log") or {}).get("encryption_input_s"),
        "weight_encryption_s": (row.get("timing_log") or {}).get("encryption_weight_s"),
        "server_only_s": row.get("server_only_s"),
        "total_s": row.get("total_s"),
        "per_sample_s": row.get("runtime_per_sample_s"),
        "semantic_validation_passed": row.get("semantic_validation_passed"),
        "relative_l2": row.get("relative_l2"),
        "sample_packing_utilization": row.get("sample_packing_utilization"),
        "input_entry_utilization_bounded": row.get("input_entry_utilization_bounded"),
        "peak_ciphertext_count_estimate": row.get("peak_ciphertext_count_estimate"),
        "exception_type": row.get("exception_type"),
        "exception": row.get("exception"),
    }


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# GL Shape Candidate Benchmark",
        "",
        "| mode | n | ok | shape | total_s | server_s | per_sample_s | sample_util | input_util | rel_l2 | exception |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        flat = flatten(row)
        lines.append(
            "| {mode} | {n} | {ok} | `{shape}` | {total} | {server} | {per} | {sample_util} | {input_util} | {rel} | {exc} |".format(
                mode=flat["mode"],
                n=flat["n_samples"],
                ok=flat["ok"],
                shape=flat["selected_shape"],
                total="" if flat["total_s"] is None else f"{flat['total_s']:.3f}",
                server="" if flat["server_only_s"] is None else f"{flat['server_only_s']:.3f}",
                per="" if flat["per_sample_s"] is None else f"{flat['per_sample_s']:.6f}",
                sample_util="" if flat["sample_packing_utilization"] is None else f"{flat['sample_packing_utilization']:.3f}",
                input_util="" if flat["input_entry_utilization_bounded"] is None else f"{flat['input_entry_utilization_bounded']:.3f}",
                rel="" if flat["relative_l2"] is None else f"{flat['relative_l2']:.3e}",
                exc=flat["exception_type"] or "",
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows: list[dict[str, Any]] = []
    for n_samples in N_SAMPLES:
        rows.append(run_one(n_samples, "fixed_256_32_32", FIXED_SHAPE))
        auto = choose_shape(n_samples, *DIMS, block_size=BLOCK_SIZE)
        rows.append(run_one(n_samples, "auto_shape", tuple(auto["shape"])))
    write_json(RESULTS / "gl_shape_candidate_benchmark.json", {"ok": True, "results": rows})
    write_csv(RESULTS / "gl_shape_candidate_benchmark.csv", [flatten(row) for row in rows])
    write_markdown(RESULTS / "gl_shape_candidate_benchmark.md", rows)
    print(RESULTS / "gl_shape_candidate_benchmark.json")


if __name__ == "__main__":
    main()
