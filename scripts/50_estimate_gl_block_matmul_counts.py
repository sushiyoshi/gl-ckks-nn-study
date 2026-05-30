from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS


BLOCK_SIZE = 32


def ceil_blocks(dim: int) -> int:
    return math.ceil(dim / BLOCK_SIZE)


def mlp_count(n_in: int, n_hidden: int, n_out: int) -> tuple[int, int, int]:
    first = ceil_blocks(n_in) * ceil_blocks(n_hidden)
    second = ceil_blocks(n_hidden) * ceil_blocks(n_out)
    return first, second, first + second


def swiglu_count(hidden: int, intermediate: int) -> tuple[int, int, int, int]:
    gate = ceil_blocks(hidden) * ceil_blocks(intermediate)
    up = gate
    down = ceil_blocks(intermediate) * ceil_blocks(hidden)
    return gate, up, down, gate + up + down


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows: list[dict[str, object]] = []
    for name, dims in [
        ("PCA32 toy", (32, 32, 10)),
        ("raw64 digits", (64, 32, 10)),
        ("original Transformer base FFN", (512, 2048, 512)),
        ("GPT-3 175B rough FFN", (12288, 49152, 12288)),
    ]:
        first, second, total = mlp_count(*dims)
        rows.append(
            {
                "model": name,
                "type": "MLP",
                "block_size": BLOCK_SIZE,
                "dims": f"{dims[0]}->{dims[1]}->{dims[2]}",
                "linear1_block_matmuls": first,
                "linear2_block_matmuls": second,
                "gate_proj_block_matmuls": "",
                "up_proj_block_matmuls": "",
                "down_proj_block_matmuls": "",
                "total_block_matmuls": total,
                "metadata": "estimate only; not a runtime benchmark",
            }
        )
    for name, hidden, intermediate in [
        ("Llama2-7B style SwiGLU", 4096, 11008),
        ("Llama3-8B style SwiGLU", 4096, 14336),
    ]:
        gate, up, down, total = swiglu_count(hidden, intermediate)
        rows.append(
            {
                "model": name,
                "type": "SwiGLU",
                "block_size": BLOCK_SIZE,
                "dims": f"hidden={hidden}, intermediate={intermediate}",
                "linear1_block_matmuls": "",
                "linear2_block_matmuls": "",
                "gate_proj_block_matmuls": gate,
                "up_proj_block_matmuls": up,
                "down_proj_block_matmuls": down,
                "total_block_matmuls": total,
                "metadata": "estimate only; not a runtime benchmark",
            }
        )
    csv_path = RESULTS / "gl_block_matmul_estimates.csv"
    md_path = RESULTS / "gl_block_matmul_estimates.md"
    write_csv(csv_path, rows)
    lines = [
        "# GL Block Matmul Count Estimates",
        "",
        "Metadata: estimates only; not a performance benchmark. Block size is 32.",
        "",
        "| Model | Type | Dims | Total block matmuls |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {row['type']} | {row['dims']} | {row['total_block_matmuls']} |")
    md_path.write_text("\n".join(lines) + "\n")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
