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


SHAPES = [
    (16, 256, 256),
    (16, 512, 512),
    (4, 1024, 1024),
    (4, 2048, 2048),
]


def run_candidate(shape: tuple[int, int, int]) -> dict[str, Any]:
    from desilofhe import GLEngine

    out: dict[str, Any] = {"ok": False, "requested_shape": list(shape), "block_size": shape[1]}
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
        plain[:, : min(8, shape[1]), : min(8, shape[2])] = 0.125
        t = now_seconds()
        ct = engine.encrypt(plain, sk)
        dec = np.asarray(engine.decrypt(ct, sk), dtype=np.float64)
        out["encrypt_decrypt_s"] = elapsed(t)
        out["encrypt_decrypt_ok"] = bool(np.allclose(plain, dec, rtol=1e-6, atol=1e-6))
        out["ciphertext_info"] = ct_info(ct)

        weight = np.zeros(shape, dtype=np.float64)
        eye = np.eye(shape[1], shape[2], dtype=np.float64)
        for batch in range(shape[0]):
            weight[batch] = eye
        t = now_seconds()
        ct_mm = engine.matrix_multiply(engine.encode(weight), ct, mm_key)
        mm_dec = np.asarray(engine.decrypt(ct_mm, sk), dtype=np.float64)
        out["matrix_multiply_s"] = elapsed(t)
        out["matrix_multiply_ok"] = bool(np.allclose(plain, mm_dec, rtol=1e-5, atol=1e-5))
        out["matrix_multiply_output"] = ct_info(ct_mm)

        coeffs = np.asarray([0.1, 0.9, 0.01, 0.001], dtype=np.float64)
        expected = coeffs[0] + coeffs[1] * plain + coeffs[2] * plain**2 + coeffs[3] * plain**3
        t = now_seconds()
        ct_poly = engine.evaluate_polynomial(ct, coeffs, had_key)
        poly_dec = np.asarray(engine.decrypt(ct_poly, sk), dtype=np.float64)
        out["evaluate_polynomial_s"] = elapsed(t)
        out["evaluate_polynomial_ok"] = bool(np.allclose(expected, poly_dec, rtol=1e-5, atol=1e-5))
        out["evaluate_polynomial_output"] = ct_info(ct_poly)
        out["ok"] = bool(out["encrypt_decrypt_ok"] and out["matrix_multiply_ok"] and out["evaluate_polynomial_ok"])
    except BaseException as exc:
        out["exception_type"] = type(exc).__name__
        out["exception"] = str(exc)
        out["traceback"] = traceback.format_exc(limit=6)
    out["total_s"] = elapsed(t_total)
    return out


def run_candidate_subprocess(shape: tuple[int, int, int], timeout_s: int) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--shape",
        ",".join(str(v) for v in shape),
    ]
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "requested_shape": list(shape),
            "block_size": shape[1],
            "exception_type": "TimeoutExpired",
            "exception": str(exc),
        }
    if completed.returncode != 0:
        return {
            "ok": False,
            "requested_shape": list(shape),
            "block_size": shape[1],
            "exception_type": "SubprocessFailed",
            "exception": completed.stderr.strip() or completed.stdout.strip(),
            "returncode": completed.returncode,
        }
    return json.loads(completed.stdout)


def parse_shape(text: str) -> tuple[int, int, int]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != 3:
        raise ValueError(text)
    return values


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": row.get("ok"),
        "requested_shape": row.get("requested_shape"),
        "engine_shape": row.get("engine_shape"),
        "block_size": row.get("block_size"),
        "max_level": row.get("max_level"),
        "slot_count": row.get("slot_count"),
        "keygen_s": row.get("keygen_s"),
        "encrypt_decrypt_s": row.get("encrypt_decrypt_s"),
        "encrypt_decrypt_ok": row.get("encrypt_decrypt_ok"),
        "matrix_multiply_s": row.get("matrix_multiply_s"),
        "matrix_multiply_ok": row.get("matrix_multiply_ok"),
        "evaluate_polynomial_s": row.get("evaluate_polynomial_s"),
        "evaluate_polynomial_ok": row.get("evaluate_polynomial_ok"),
        "total_s": row.get("total_s"),
        "exception_type": row.get("exception_type"),
        "exception": row.get("exception"),
    }


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Large GLEngine Shape Probe",
        "",
        "| ok | shape | keygen_s | encdec_ok | mm_ok | poly_ok | total_s | exception |",
        "|---|---|---:|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {ok} | `{shape}` | {keygen} | {enc} | {mm} | {poly} | {total} | {exc} |".format(
                ok=row.get("ok"),
                shape=row.get("requested_shape"),
                keygen="" if row.get("keygen_s") is None else f"{row.get('keygen_s'):.3f}",
                enc=row.get("encrypt_decrypt_ok"),
                mm=row.get("matrix_multiply_ok"),
                poly=row.get("evaluate_polynomial_ok"),
                total="" if row.get("total_s") is None else f"{row.get('total_s'):.3f}",
                exc=row.get("exception_type") or "",
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shape")
    parser.add_argument("--timeout-s", type=int, default=300)
    args = parser.parse_args()

    if args.worker:
        if not args.shape:
            raise SystemExit("--worker requires --shape")
        print(json.dumps(run_candidate(parse_shape(args.shape)), default=str))
        return

    rows = [run_candidate_subprocess(shape, args.timeout_s) for shape in SHAPES]
    write_json(RESULTS / "gl_large_shape_probe.json", {"ok": True, "results": rows})
    write_csv(RESULTS / "gl_large_shape_probe.csv", [flatten(row) for row in rows])
    write_markdown(RESULTS / "gl_large_shape_probe.md", rows)
    print(RESULTS / "gl_large_shape_probe.json")


if __name__ == "__main__":
    main()
