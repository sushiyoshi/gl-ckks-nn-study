from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, ct_info, elapsed, now_seconds, write_csv, write_json


BATCH_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BLOCK_SIZES = [16, 32, 64]


def run_candidate(batch_count: int, block_size: int) -> dict[str, Any]:
    from desilofhe import GLEngine

    shape = (batch_count, block_size, block_size)
    out: dict[str, Any] = {
        "ok": False,
        "batch_count": batch_count,
        "block_size": block_size,
        "requested_shape": list(shape),
    }
    t_total = now_seconds()
    try:
        t = now_seconds()
        engine = GLEngine(shape=shape)
        out["engine_create_s"] = elapsed(t)
        out["engine_shape"] = list(tuple(int(v) for v in engine.shape))
        out["max_level"] = getattr(engine, "max_level", None)
        out["slot_count"] = getattr(engine, "slot_count", None)

        t = now_seconds()
        sk = engine.create_secret_key()
        mm_key = engine.create_matrix_multiplication_key(sk)
        had_key = engine.create_hadamard_multiplication_key(sk)
        out["keygen_s"] = elapsed(t)

        plain = np.zeros(shape, dtype=np.float64)
        plain[:, : min(4, block_size), : min(4, block_size)] = 0.25
        t = now_seconds()
        ct = engine.encrypt(plain, sk)
        dec = np.asarray(engine.decrypt(ct, sk), dtype=np.float64)
        out["encrypt_decrypt_s"] = elapsed(t)
        out["encrypt_decrypt_ok"] = bool(np.allclose(plain, dec, rtol=1e-6, atol=1e-6))
        out["ciphertext_info"] = ct_info(ct)

        weight = np.zeros(shape, dtype=np.float64)
        for batch in range(batch_count):
            weight[batch, :block_size, :block_size] = np.eye(block_size)
        t = now_seconds()
        ct_mm = engine.matrix_multiply(engine.encode(weight), ct, mm_key)
        mm_dec = np.asarray(engine.decrypt(ct_mm, sk), dtype=np.float64)
        out["matrix_multiply_s"] = elapsed(t)
        out["matrix_multiply_ok"] = bool(np.allclose(plain, mm_dec, rtol=1e-5, atol=1e-5))
        out["matrix_multiply_output"] = ct_info(ct_mm)

        t = now_seconds()
        coeffs = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        ct_poly = engine.evaluate_polynomial(ct, coeffs, had_key)
        poly_dec = np.asarray(engine.decrypt(ct_poly, sk), dtype=np.float64)
        out["evaluate_polynomial_s"] = elapsed(t)
        out["evaluate_polynomial_ok"] = bool(np.allclose(plain, poly_dec, rtol=1e-5, atol=1e-5))
        out["evaluate_polynomial_output"] = ct_info(ct_poly)
        out["ok"] = bool(out["encrypt_decrypt_ok"] and out["matrix_multiply_ok"] and out["evaluate_polynomial_ok"])
    except BaseException as exc:
        out["exception_type"] = type(exc).__name__
        out["exception"] = str(exc)
        out["traceback"] = traceback.format_exc(limit=6)
    out["total_s"] = elapsed(t_total)
    return out


def run_candidate_subprocess(batch_count: int, block_size: int, timeout_s: int) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--batch-count",
        str(batch_count),
        "--block-size",
        str(block_size),
    ]
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "batch_count": batch_count,
            "block_size": block_size,
            "requested_shape": [batch_count, block_size, block_size],
            "exception_type": "TimeoutExpired",
            "exception": str(exc),
        }
    if completed.returncode != 0:
        return {
            "ok": False,
            "batch_count": batch_count,
            "block_size": block_size,
            "requested_shape": [batch_count, block_size, block_size],
            "exception_type": "SubprocessFailed",
            "exception": completed.stderr.strip() or completed.stdout.strip(),
            "returncode": completed.returncode,
        }
    return json.loads(completed.stdout)


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": row.get("ok"),
        "batch_count": row.get("batch_count"),
        "block_size": row.get("block_size"),
        "requested_shape": row.get("requested_shape"),
        "engine_shape": row.get("engine_shape"),
        "max_level": row.get("max_level"),
        "slot_count": row.get("slot_count"),
        "engine_create_s": row.get("engine_create_s"),
        "keygen_s": row.get("keygen_s"),
        "encrypt_decrypt_ok": row.get("encrypt_decrypt_ok"),
        "matrix_multiply_ok": row.get("matrix_multiply_ok"),
        "evaluate_polynomial_ok": row.get("evaluate_polynomial_ok"),
        "total_s": row.get("total_s"),
        "exception_type": row.get("exception_type"),
        "exception": row.get("exception"),
    }


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# GLEngine Shape Probe",
        "",
        "| ok | B | block | requested_shape | engine_shape | keygen_s | mm_ok | poly_ok | exception |",
        "|---|---:|---:|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {ok} | {b} | {d} | `{req}` | `{eng}` | {keygen} | {mm} | {poly} | {exc} |".format(
                ok=row.get("ok"),
                b=row.get("batch_count"),
                d=row.get("block_size"),
                req=row.get("requested_shape"),
                eng=row.get("engine_shape"),
                keygen="" if row.get("keygen_s") is None else f"{row.get('keygen_s'):.3f}",
                mm=row.get("matrix_multiply_ok"),
                poly=row.get("evaluate_polynomial_ok"),
                exc=(row.get("exception_type") or ""),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--batch-count", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--timeout-s", type=int, default=120)
    args = parser.parse_args()

    if args.worker:
        if args.batch_count is None or args.block_size is None:
            raise SystemExit("--worker requires --batch-count and --block-size")
        print(json.dumps(run_candidate(args.batch_count, args.block_size), default=str))
        return

    rows = [
        run_candidate_subprocess(batch_count, block_size, args.timeout_s)
        for block_size in BLOCK_SIZES
        for batch_count in BATCH_COUNTS
    ]
    payload = {"ok": True, "candidate_count": len(rows), "results": rows}
    write_json(RESULTS / "glengine_shape_probe.json", payload)
    write_csv(RESULTS / "glengine_shape_probe.csv", [flatten(row) for row in rows])
    write_markdown(RESULTS / "glengine_shape_probe.md", rows)
    print(RESULTS / "glengine_shape_probe.json")


if __name__ == "__main__":
    main()
