from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_block_schedule import ffn_block_matmuls, swiglu_block_matmuls
from src.logging_utils import RESULTS, read_json, write_csv


def observed_block_times() -> dict[str, float | None]:
    raw64_path = RESULTS / "gl_encrypted_weight_raw64_mlp_n8192.json"
    pca_path = RESULTS / "gl_encrypted_weight_pca32_mlp_n8192.json"
    out: dict[str, float | None] = {"raw64_per_block_s": None, "pca32_per_block_s": None}
    if raw64_path.exists():
        raw = read_json(raw64_path)
        if raw.get("server_only_s") is not None and raw.get("block_matmul_count"):
            out["raw64_per_block_s"] = raw["server_only_s"] / raw["block_matmul_count"]
    if pca_path.exists():
        pca = read_json(pca_path)
        count = pca.get("block_matmul_count") or 2
        if pca.get("server_only_s") is not None:
            out["pca32_per_block_s"] = pca["server_only_s"] / count
    return out


def estimate(value: float | None, count: int) -> float | None:
    return None if value is None else value * count


def main() -> None:
    times = observed_block_times()
    rows = []
    specs = [
        ("PCA32 toy", "FFN", (32, 32, 10)),
        ("raw64 digits", "FFN", (64, 32, 10)),
        ("synthetic medium", "FFN", (256, 512, 256)),
        ("original Transformer FFN", "FFN", (512, 2048, 512)),
        ("Llama2-7B style SwiGLU", "SwiGLU", (4096, 11008)),
        ("Llama3-8B style SwiGLU", "SwiGLU", (4096, 14336)),
    ]
    for name, kind, dims in specs:
        if kind == "FFN":
            sched = ffn_block_matmuls(*dims)
            dim_text = f"{dims[0]}->{dims[1]}->{dims[2]}"
            row = {
                "model": name,
                "kind": kind,
                "dims": dim_text,
                **sched,
                "estimated_server_only_s_raw64_block_time": estimate(times["raw64_per_block_s"], sched["total_block_matmuls"]),
                "estimated_server_only_s_pca32_block_time": estimate(times["pca32_per_block_s"], sched["total_block_matmuls"]),
            }
        else:
            sched = swiglu_block_matmuls(*dims)
            dim_text = f"hidden={dims[0]}, intermediate={dims[1]}"
            row = {
                "model": name,
                "kind": kind,
                "dims": dim_text,
                **sched,
                "estimated_server_only_s_raw64_block_time": estimate(times["raw64_per_block_s"], sched["total_block_matmuls"]),
                "estimated_server_only_s_pca32_block_time": estimate(times["pca32_per_block_s"], sched["total_block_matmuls"]),
            }
        rows.append(row)

    csv_path = RESULTS / "gl_transformer_ffn_schedule_estimates.csv"
    md_path = RESULTS / "gl_transformer_ffn_schedule_estimates.md"
    write_csv(csv_path, rows)
    headers = [
        "model",
        "kind",
        "dims",
        "input_blocks",
        "hidden_blocks",
        "intermediate_blocks",
        "output_blocks",
        "total_block_matmuls",
        "activation_blocks",
        "estimated_server_only_s_raw64_block_time",
        "estimated_server_only_s_pca32_block_time",
    ]
    lines = [
        "# GL Transformer FFN block schedule estimates",
        "",
        "These are simple linear extrapolations from observed per-block times. They do not include memory, cache, level, bootstrapping, scheduling, or key/material pressure effects.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)
    print(csv_path)


if __name__ == "__main__":
    main()
