from __future__ import annotations

import math
from typing import Any

import numpy as np


def split_features_to_blocks(samples: np.ndarray, block_size: int = 32) -> list[np.ndarray]:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"samples must be 2D, got {x.shape}")
    blocks: list[np.ndarray] = []
    for start in range(0, x.shape[1], block_size):
        chunk = x[:, start : start + block_size]
        block = np.zeros((x.shape[0], block_size), dtype=np.float64)
        block[:, : chunk.shape[1]] = chunk
        blocks.append(block)
    if not blocks:
        blocks.append(np.zeros((x.shape[0], block_size), dtype=np.float64))
    return blocks


def pack_sample_columns_for_blocks(
    blocks: list[np.ndarray],
    shape: tuple[int, int, int] = (256, 32, 32),
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if not blocks:
        raise ValueError("at least one block is required")
    n_samples = int(blocks[0].shape[0])
    block_size = int(blocks[0].shape[1])
    batches, rows, cols = shape
    if block_size > rows:
        raise ValueError(f"block_size {block_size} exceeds GL rows {rows}")
    used_batches = math.ceil(n_samples / cols) if n_samples else 0
    if used_batches > batches:
        raise ValueError(f"{n_samples} samples need {used_batches} batches, exceeds GL batch axis {batches}")

    placements = [
        {"sample_index": i, "batch": i // cols, "col": i % cols}
        for i in range(n_samples)
    ]
    tensors: list[np.ndarray] = []
    for block_index, block in enumerate(blocks):
        b = np.asarray(block, dtype=np.float64)
        if b.shape != (n_samples, block_size):
            raise ValueError(f"block {block_index} shape {b.shape} does not match {(n_samples, block_size)}")
        tensor = np.zeros(shape, dtype=np.float64)
        for item in placements:
            tensor[item["batch"], :block_size, item["col"]] = b[item["sample_index"]]
        tensors.append(tensor)

    layout = {
        "n_samples": n_samples,
        "block_size": block_size,
        "max_cols": cols,
        "used_batches": used_batches,
        "used_columns_last_batch": 0 if n_samples == 0 else ((n_samples - 1) % cols) + 1,
        "samples_per_ciphertext_capacity": batches * cols,
        "placements": placements,
    }
    return tensors, layout


def split_weight_matrix_for_column_vector_eval(
    weight: np.ndarray,
    n_in: int,
    n_out: int,
    block_size: int = 32,
) -> list[list[np.ndarray]]:
    w = np.asarray(weight, dtype=np.float64)
    if w.shape != (n_in, n_out):
        raise ValueError(f"weight shape {w.shape} does not match {(n_in, n_out)}")
    in_blocks = math.ceil(n_in / block_size)
    out_blocks = math.ceil(n_out / block_size)
    w_t = w.T
    blocks: list[list[np.ndarray]] = []
    for out_block in range(out_blocks):
        row_start = out_block * block_size
        row_end = min(row_start + block_size, n_out)
        row_blocks: list[np.ndarray] = []
        for in_block in range(in_blocks):
            col_start = in_block * block_size
            col_end = min(col_start + block_size, n_in)
            block = np.zeros((block_size, block_size), dtype=np.float64)
            block[: row_end - row_start, : col_end - col_start] = w_t[row_start:row_end, col_start:col_end]
            row_blocks.append(block)
        blocks.append(row_blocks)
    return blocks


def broadcast_weight_blocks_to_gl(
    weight_blocks: list[list[np.ndarray]],
    shape: tuple[int, int, int],
    used_batches: int,
) -> list[list[np.ndarray]]:
    tensors: list[list[np.ndarray]] = []
    for out_blocks in weight_blocks:
        row: list[np.ndarray] = []
        for block in out_blocks:
            w = np.asarray(block, dtype=np.float64)
            if w.shape[0] > shape[1] or w.shape[1] > shape[2]:
                raise ValueError(f"weight block {w.shape} exceeds GL matrix {shape[1:]}")
            tensor = np.zeros(shape, dtype=np.float64)
            for batch in range(used_batches):
                tensor[batch, : w.shape[0], : w.shape[1]] = w
            row.append(tensor)
        tensors.append(row)
    return tensors


def bias_block_tensor(
    bias: np.ndarray,
    out_block_index: int,
    layout: dict[str, Any],
    shape: tuple[int, int, int],
    n_out: int,
    block_size: int = 32,
) -> np.ndarray:
    b = np.asarray(bias, dtype=np.float64).ravel()
    if b.shape[0] != n_out:
        raise ValueError(f"bias shape {b.shape} does not match n_out={n_out}")
    start = out_block_index * block_size
    end = min(start + block_size, n_out)
    block = np.zeros(block_size, dtype=np.float64)
    block[: end - start] = b[start:end]
    tensor = np.zeros(shape, dtype=np.float64)
    for item in layout["placements"]:
        tensor[item["batch"], :block_size, item["col"]] = block
    return tensor


def unpack_output_blocks(
    blocks: list[np.ndarray],
    layout: dict[str, Any],
    n_out: int,
    block_size: int = 32,
) -> np.ndarray:
    out = np.zeros((layout["n_samples"], n_out), dtype=np.float64)
    for out_block_index, block in enumerate(blocks):
        start = out_block_index * block_size
        end = min(start + block_size, n_out)
        if start >= n_out:
            break
        arr = np.asarray(block, dtype=np.float64)
        for item in layout["placements"]:
            out[item["sample_index"], start:end] = arr[item["batch"], : end - start, item["col"]]
    return out


def block_linear_plain_gl_layout(
    input_blocks: list[np.ndarray],
    weight_blocks: list[list[np.ndarray]],
    bias: np.ndarray,
    n_out: int,
    layout: dict[str, Any],
    shape: tuple[int, int, int],
    block_size: int = 32,
) -> tuple[list[np.ndarray], np.ndarray]:
    out_tensors: list[np.ndarray] = []
    for out_block_index, row in enumerate(weight_blocks):
        acc = np.zeros(shape, dtype=np.float64)
        for in_block_index, weight_block in enumerate(row):
            w_tensor = np.zeros(shape, dtype=np.float64)
            for batch in range(layout["used_batches"]):
                w_tensor[batch, :block_size, :block_size] = weight_block
            acc += np.matmul(w_tensor, input_blocks[in_block_index])
        acc += bias_block_tensor(bias, out_block_index, layout, shape, n_out, block_size)
        out_tensors.append(acc)
    return out_tensors, unpack_output_blocks(out_tensors, layout, n_out, block_size)


def block_linear_encrypted(
    engine: Any,
    ct_input_blocks: list[Any],
    ct_weight_blocks: list[list[Any]],
    bias_plain_blocks: list[Any],
    mm_key: Any,
    level_log: list[dict[str, Any]] | None = None,
    timing_log: dict[str, float] | None = None,
    stage_prefix: str = "linear",
    timer: Any | None = None,
    elapsed_fn: Any | None = None,
    ct_info_fn: Any | None = None,
) -> list[Any]:
    outputs: list[Any] = []
    for out_block_index, row in enumerate(ct_weight_blocks):
        acc = None
        for in_block_index, ct_w in enumerate(row):
            start = timer() if timer else None
            prod = engine.matrix_multiply(ct_w, ct_input_blocks[in_block_index], mm_key)
            key = f"{stage_prefix}_out{out_block_index}_in{in_block_index}_matrix_multiply_s"
            if timing_log is not None and start is not None and elapsed_fn is not None:
                timing_log[key] = elapsed_fn(start)
            if level_log is not None and ct_info_fn is not None:
                level_log.append({"stage": f"{stage_prefix}_out{out_block_index}_in{in_block_index}", **ct_info_fn(prod)})
            acc = prod if acc is None else engine.add(acc, prod)
        if acc is None:
            raise ValueError("empty weight row")
        acc = engine.add(acc, bias_plain_blocks[out_block_index])
        if level_log is not None and ct_info_fn is not None:
            level_log.append({"stage": f"{stage_prefix}_out{out_block_index}_sum", **ct_info_fn(acc)})
        outputs.append(acc)
    return outputs


def block_linear_plain_weight_encrypted_input(
    engine: Any,
    ct_input_blocks: list[Any],
    pt_weight_blocks: list[list[Any]],
    bias_plain_blocks: list[Any],
    mm_key: Any,
    level_log: list[dict[str, Any]] | None = None,
    timing_log: dict[str, float] | None = None,
    stage_prefix: str = "linear",
    timer: Any | None = None,
    elapsed_fn: Any | None = None,
    ct_info_fn: Any | None = None,
) -> list[Any]:
    outputs: list[Any] = []
    for out_block_index, row in enumerate(pt_weight_blocks):
        acc = None
        for in_block_index, pt_w in enumerate(row):
            start = timer() if timer else None
            prod = engine.matrix_multiply(pt_w, ct_input_blocks[in_block_index], mm_key)
            key = f"{stage_prefix}_out{out_block_index}_in{in_block_index}_matrix_multiply_s"
            if timing_log is not None and start is not None and elapsed_fn is not None:
                timing_log[key] = elapsed_fn(start)
            if level_log is not None and ct_info_fn is not None:
                level_log.append({"stage": f"{stage_prefix}_out{out_block_index}_in{in_block_index}", **ct_info_fn(prod)})
            acc = prod if acc is None else engine.add(acc, prod)
        if acc is None:
            raise ValueError("empty weight row")
        acc = engine.add(acc, bias_plain_blocks[out_block_index])
        if level_log is not None and ct_info_fn is not None:
            level_log.append({"stage": f"{stage_prefix}_out{out_block_index}_sum", **ct_info_fn(acc)})
        outputs.append(acc)
    return outputs


def apply_polynomial_to_blocks(
    engine: Any,
    ct_blocks: list[Any],
    coeffs: np.ndarray,
    had_key: Any,
    level_log: list[dict[str, Any]] | None = None,
    timing_log: dict[str, float] | None = None,
    timer: Any | None = None,
    elapsed_fn: Any | None = None,
    ct_info_fn: Any | None = None,
    stage_prefix: str = "activation",
) -> list[Any]:
    out: list[Any] = []
    for block_index, ct in enumerate(ct_blocks):
        start = timer() if timer else None
        ct_out = engine.evaluate_polynomial(ct, coeffs, had_key)
        if timing_log is not None and start is not None and elapsed_fn is not None:
            timing_log[f"{stage_prefix}_block{block_index}_s"] = elapsed_fn(start)
        if level_log is not None and ct_info_fn is not None:
            level_log.append({"stage": f"{stage_prefix}_block{block_index}", **ct_info_fn(ct_out)})
        out.append(ct_out)
    return out
