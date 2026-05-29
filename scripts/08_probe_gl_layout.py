from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, ct_info, elapsed, exception_record, now_seconds, write_json
from src.metrics import error_metrics


def make_probe_arrays(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    batches, rows, cols = shape
    if batches < 2 or rows < 3 or cols < 3:
        raise ValueError(f"Probe requires at least shape (2, 3, 3), got {shape}")
    a = np.zeros(shape, dtype=np.float64)
    b = np.zeros(shape, dtype=np.float64)

    a[0, :2, :3] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b[0, :3, :1] = np.array([[7.0], [8.0], [9.0]])

    a[1, :2, :3] = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    b[1, :3, :1] = np.array([[0.5], [1.5], [2.5]])
    return a, b


def small_region(x: np.ndarray, batch: int, rows: int = 4, cols: int = 4) -> list[list[float]]:
    return np.asarray(x[batch, :rows, :cols], dtype=np.float64).tolist()


def close_summary(expected: np.ndarray, actual: np.ndarray, prefix: str) -> dict[str, Any]:
    region_expected = expected[:2, :1]
    region_actual = actual[:2, :1]
    return {
        "expected_top_2x1": region_expected.tolist(),
        "actual_top_2x1": region_actual.tolist(),
        "allclose_top_2x1": bool(np.allclose(region_expected, region_actual, rtol=1e-5, atol=1e-5)),
        **error_metrics(region_expected, region_actual, prefix),
    }


def run_matrix_case(engine, sk, mm_key, name: str, lhs: Any, rhs: Any, expected: np.ndarray) -> dict[str, Any]:
    try:
        start = now_seconds()
        out = engine.matrix_multiply(lhs, rhs, mm_key)
        multiply_s = elapsed(start)
        start = now_seconds()
        dec = np.asarray(engine.decrypt(out, sk), dtype=np.float64)
        decrypt_s = elapsed(start)
        batches = {}
        for batch in [0, 1]:
            batches[str(batch)] = {
                "decrypted_region_4x4": small_region(dec, batch),
                **close_summary(expected[batch], dec[batch], f"{name}_batch{batch}"),
            }
        return {
            "ok": True,
            "multiply_s": multiply_s,
            "decrypt_s": decrypt_s,
            "ciphertext": ct_info(out),
            "batches": batches,
        }
    except Exception as exc:
        return exception_record(exc, case=name)


def run_toy_32x32_case(engine, sk, mm_key, name: str, weights: Any, inputs: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    try:
        ct_inputs = engine.encrypt(inputs, sk)
        start = now_seconds()
        out = engine.matrix_multiply(weights, ct_inputs, mm_key)
        multiply_s = elapsed(start)
        start = now_seconds()
        dec = np.asarray(engine.decrypt(out, sk), dtype=np.float64)
        decrypt_s = elapsed(start)
        batches = {}
        for batch in [0, 1]:
            batches[str(batch)] = {
                "expected_first_8": expected[batch, :8, 0].tolist(),
                "actual_first_8": dec[batch, :8, 0].tolist(),
                "allclose_32x1": bool(np.allclose(expected[batch, :32, :1], dec[batch, :32, :1], rtol=1e-5, atol=1e-5)),
                **error_metrics(expected[batch, :32, :1], dec[batch, :32, :1], f"{name}_batch{batch}"),
            }
        return {
            "ok": True,
            "multiply_s": multiply_s,
            "decrypt_s": decrypt_s,
            "ciphertext": ct_info(out),
            "batches": batches,
        }
    except Exception as exc:
        return exception_record(exc, case=name)


def run_toy_32x32_linear(engine, sk, mm_key, shape: tuple[int, int, int]) -> dict[str, Any]:
    if shape[1] < 32 or shape[2] < 32:
        return {"ok": False, "reason": f"engine shape {shape} cannot hold 32x32 toy matrices"}

    weights = np.zeros(shape, dtype=np.float64)
    inputs = np.zeros(shape, dtype=np.float64)
    base_w = (np.arange(32 * 32, dtype=np.float64).reshape(32, 32) % 11) / 10.0
    base_x = np.linspace(-1.0, 1.0, 32, dtype=np.float64).reshape(32, 1)

    weights[0, :32, :32] = base_w
    inputs[0, :32, :1] = base_x
    weights[1, :32, :32] = base_w.T
    inputs[1, :32, :1] = base_x[::-1]
    expected = np.matmul(weights, inputs)

    return {
        "ok": True,
        "description": "32->32 toy linear layer as batched 32x32 plaintext weight times encrypted 32x1 input column.",
        "raw_numpy_weight_times_ct_input": run_toy_32x32_case(engine, sk, mm_key, "toy32_raw_numpy_weight", weights, inputs, expected),
        "encoded_glplaintext_weight_times_ct_input": run_toy_32x32_case(engine, sk, mm_key, "toy32_encoded_weight", engine.encode(weights), inputs, expected),
    }


def main() -> None:
    from desilofhe import GLEngine

    payload: dict[str, Any] = {}
    try:
        engine = GLEngine()
        shape = tuple(int(x) for x in getattr(engine, "shape"))
        payload["engine_shape"] = shape
        payload["slot_count"] = int(np.prod(shape))
        payload["max_level"] = getattr(engine, "max_level", None)

        a, b = make_probe_arrays(shape)
        expected = np.matmul(a, b)
        payload["plain_expected"] = {
            "batch0_A_region_4x4": small_region(a, 0),
            "batch0_B_region_4x4": small_region(b, 0),
            "batch0_A_at_B_region_4x4": small_region(expected, 0),
            "batch1_A_region_4x4": small_region(a, 1),
            "batch1_B_region_4x4": small_region(b, 1),
            "batch1_A_at_B_region_4x4": small_region(expected, 1),
        }

        pt_a = engine.encode(a)
        pt_b = engine.encode(b)
        dec_a = np.asarray(engine.decode(pt_a), dtype=np.float64)
        dec_b = np.asarray(engine.decode(pt_b), dtype=np.float64)
        payload["encode_decode"] = {
            "A_shape": list(dec_a.shape),
            "B_shape": list(dec_b.shape),
            "A_allclose": bool(np.allclose(a, dec_a)),
            "B_allclose": bool(np.allclose(b, dec_b)),
            "A_batch0_region_4x4": small_region(dec_a, 0),
            "B_batch0_region_4x4": small_region(dec_b, 0),
            "A_batch1_region_4x4": small_region(dec_a, 1),
            "B_batch1_region_4x4": small_region(dec_b, 1),
        }

        start = now_seconds()
        sk = engine.create_secret_key()
        mm_key = engine.create_matrix_multiplication_key(sk)
        payload["key_generation_s"] = elapsed(start)

        ct_a = engine.encrypt(a, sk)
        ct_b = engine.encrypt(b, sk)
        payload["encrypted_inputs"] = {"A": ct_info(ct_a), "B": ct_info(ct_b)}
        payload["matrix_multiply_cases"] = {
            "plain_A_times_ct_B": run_matrix_case(engine, sk, mm_key, "plain_A_times_ct_B", a, ct_b, expected),
            "pt_A_times_ct_B": run_matrix_case(engine, sk, mm_key, "pt_A_times_ct_B", pt_a, ct_b, expected),
            "ct_A_times_plain_B": run_matrix_case(engine, sk, mm_key, "ct_A_times_plain_B", ct_a, b, expected),
            "ct_A_times_pt_B": run_matrix_case(engine, sk, mm_key, "ct_A_times_pt_B", ct_a, pt_b, expected),
            "ct_A_times_ct_B": run_matrix_case(engine, sk, mm_key, "ct_A_times_ct_B", ct_a, ct_b, expected),
        }
        payload["toy_32x32_linear"] = run_toy_32x32_linear(engine, sk, mm_key, shape)
        cases = payload["matrix_multiply_cases"]
        toy = payload["toy_32x32_linear"]
        payload["layout_conclusion"] = {
            "shape_semantics": "engine.shape is interpreted as (batch, rows, cols); matrix_multiply performs independent matrix products over the last two axes.",
            "encode_decode_preserves_shape": payload["encode_decode"]["A_allclose"] and payload["encode_decode"]["B_allclose"],
            "raw_numpy_left_operand_matches_matmul": all(
                cases["plain_A_times_ct_B"]["batches"][str(batch)]["allclose_top_2x1"] for batch in [0, 1]
            ),
            "encoded_plaintext_left_operand_matches_matmul": all(
                cases["pt_A_times_ct_B"]["batches"][str(batch)]["allclose_top_2x1"] for batch in [0, 1]
            ),
            "raw_numpy_right_operand_matches_matmul": all(
                cases["ct_A_times_plain_B"]["batches"][str(batch)]["allclose_top_2x1"] for batch in [0, 1]
            ),
            "encoded_plaintext_right_operand_matches_matmul": all(
                cases["ct_A_times_pt_B"]["batches"][str(batch)]["allclose_top_2x1"] for batch in [0, 1]
            ),
            "ct_ct_matches_matmul": all(
                cases["ct_A_times_ct_B"]["batches"][str(batch)]["allclose_top_2x1"] for batch in [0, 1]
            ),
            "toy_32x32_encoded_weight_matches": all(
                toy["encoded_glplaintext_weight_times_ct_input"]["batches"][str(batch)]["allclose_32x1"] for batch in [0, 1]
            ),
            "toy_32x32_raw_numpy_weight_matches": all(
                toy["raw_numpy_weight_times_ct_input"]["batches"][str(batch)]["allclose_32x1"] for batch in [0, 1]
            ),
        }
    except Exception as exc:
        payload["ok"] = False
        payload["error"] = exception_record(exc, case="gl_layout_probe_setup")

    write_json(RESULTS / "gl_layout_probe.json", payload)
    print(RESULTS / "gl_layout_probe.json")


if __name__ == "__main__":
    main()
