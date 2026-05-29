from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ckks_packing import make_block_diagonal_linear_matrix, make_repeated_bias_vector
from src.gl_layout import make_gl_bias_column, make_gl_column, make_gl_weight_matrix
from src.logging_utils import RESULTS, write_csv, write_json
from src.model import load_model
from src.model_pca32 import load_pca32_model
from src.polynomial import fit_relu_power_polynomial


def size_record(
    scheme: str,
    object_name: str,
    *,
    serialized_nbytes: int | None = None,
    python_object_size: int | None = None,
    supported: bool = True,
    method: str = "",
    notes: str = "",
    exception: str | None = None,
) -> dict[str, Any]:
    return {
        "scheme": scheme,
        "object_name": object_name,
        "supported": supported,
        "method": method,
        "serialized_nbytes": serialized_nbytes,
        "python_object_size": python_object_size,
        "notes": notes,
        "exception": exception,
    }


def try_serialize(engine, method_name: str, obj) -> tuple[int | None, str | None]:
    method = getattr(engine, method_name)
    try:
        data = method(obj)
        return len(data), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    from desilofhe import Engine, GLEngine

    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"objects": []}

    pca_model, pca_arrays = load_pca32_model()
    digits_model, digits_arrays = load_model()
    coeffs = np.load(Path("data") / "relu_poly_degree3.npz")["coeffs"].astype(np.float64)
    pca_radius = float(max(3.0, min(8.0, np.ceil(np.quantile(np.abs(pca_model.relu_forward(pca_arrays["x_train_32"])["z1"]), 0.995)))))

    # CKKS
    try:
        e = Engine()
        slots = int(e.slot_count)
        sk = e.create_secret_key()
        rk = e.create_rotation_key(sk)
        relk = e.create_relinearization_key(sk)
        mmk = e.create_matrix_multiplication_key(sk)
        x = pca_arrays["x_test_32"][0]
        ct = e.encrypt(np.pad(x, (0, slots - x.size)), sk)
        w1_dense = make_block_diagonal_linear_matrix(pca_model.w1.T, 32, 1, slots, 32, 32)
        w2_dense = make_block_diagonal_linear_matrix(pca_model.w2.T, 32, 1, slots, 10, 32)
        pm1 = e.encode_to_plain_matrix(w1_dense)
        pm2 = e.encode_to_plain_matrix(w2_dense)
        lm1 = e.encode_to_light_plain_matrix(w1_dense)
        lm2 = e.encode_to_light_plain_matrix(w2_dense)
        ckks_objects = [
            ("secret_key", sk, "serialize_secret_key", "CKKS secret key"),
            ("rotation_key", rk, "serialize_rotation_key", "CKKS rotation key"),
            ("relinearization_key", relk, "serialize_relinearization_key", "CKKS relinearization key"),
            ("matrix_multiplication_key", mmk, "serialize_matrix_multiplication_key", "CKKS matrix multiplication key"),
            ("ciphertext", ct, "serialize_ciphertext", "CKKS ciphertext"),
            ("plainmatrix_w1", pm1, "serialize_plain_matrix", "CKKS PlainMatrix encoded W1"),
            ("plainmatrix_w2", pm2, "serialize_plain_matrix", "CKKS PlainMatrix encoded W2"),
            ("lightplainmatrix_w1", lm1, "serialize_light_plain_matrix", "CKKS LightPlainMatrix encoded W1"),
            ("lightplainmatrix_w2", lm2, "serialize_light_plain_matrix", "CKKS LightPlainMatrix encoded W2"),
        ]
        for name, obj, ser, note in ckks_objects:
            nbytes, exc = try_serialize(e, ser, obj)
            rows.append(size_record("CKKS", name, serialized_nbytes=nbytes, python_object_size=sys.getsizeof(obj), supported=exc is None, method=ser, notes=note, exception=exc))
    except Exception as exc:
        rows.append(size_record("CKKS", "setup", supported=False, exception=f"{type(exc).__name__}: {exc}", notes="CKKS setup failed"))

    # GL
    try:
        g = GLEngine()
        sk = g.create_secret_key()
        mmk = g.create_matrix_multiplication_key(sk)
        had = g.create_hadamard_multiplication_key(sk)
        ct = g.encrypt(np.ones(g.shape, dtype=np.float64), sk)
        w1 = g.encode(make_gl_weight_matrix(pca_model.w1.T, g.shape, batch=0))
        w2 = g.encode(make_gl_weight_matrix(pca_model.w2.T, g.shape, batch=0))
        # GL has no exposed serialization API in this build; record approximate Python sizes instead.
        gl_objects = [
            ("secret_key", sk, "n/a", "GL secret key"),
            ("matrix_multiplication_key", mmk, "n/a", "GL matrix multiplication key"),
            ("hadamard_multiplication_key", had, "n/a", "GL hadamard multiplication key"),
            ("ciphertext", ct, "n/a", "GL ciphertext"),
            ("glplaintext_w1", w1, "n/a", "GLPlaintext encoded W1"),
            ("glplaintext_w2", w2, "n/a", "GLPlaintext encoded W2"),
        ]
        for name, obj, method, note in gl_objects:
            rows.append(size_record("GL", name, serialized_nbytes=None, python_object_size=sys.getsizeof(obj), supported=False, method=method, notes=note, exception="No serialize_* API exposed by GLEngine"))
    except Exception as exc:
        rows.append(size_record("GL", "setup", supported=False, exception=f"{type(exc).__name__}: {exc}", notes="GL setup failed"))

    payload["summary"] = {
        "ckks_rows": sum(1 for r in rows if r["scheme"] == "CKKS"),
        "gl_rows": sum(1 for r in rows if r["scheme"] == "GL"),
    }
    payload["objects"] = rows
    write_json(RESULTS / "object_sizes.json", payload)
    write_csv(RESULTS / "object_sizes.csv", rows)
    print(RESULTS / "object_sizes.json")


if __name__ == "__main__":
    main()
