from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ckks_packing import dense_matrix_memory_stats
from src.logging_utils import RESULTS, ct_info, elapsed, exception_record, now_seconds, write_json


def obj_info(obj) -> dict:
    return {
        "type": type(obj).__name__,
        "nbytes": getattr(obj, "nbytes", None),
        "serialized_nbytes": getattr(obj, "serialized_nbytes", None),
    }


def try_case(name: str, fn) -> dict:
    try:
        t = now_seconds()
        out = fn()
        return {"ok": True, "elapsed_s": elapsed(t), **(out if isinstance(out, dict) else {"result": str(out)})}
    except Exception as exc:
        return exception_record(exc, case=name)


def main() -> None:
    from desilofhe import Engine

    payload = {}
    engine = Engine()
    slots = int(engine.slot_count)
    payload["engine"] = {"slot_count": slots, "max_level": getattr(engine, "max_level", None)}
    payload["docstrings"] = {
        "multiply_matrix": Engine.multiply_matrix.__doc__,
        "encode_to_plain_matrix": Engine.encode_to_plain_matrix.__doc__,
        "encode_to_light_plain_matrix": Engine.encode_to_light_plain_matrix.__doc__,
    }
    matrix = np.eye(slots, dtype=np.float64)
    matrix[0, 1] = 2.0
    payload["dense_identity_plus_one"] = dense_matrix_memory_stats(matrix)
    sk = engine.create_secret_key()
    rot_key = engine.create_rotation_key(sk)
    mm_key = engine.create_matrix_multiplication_key(sk)
    ct = engine.encrypt(np.arange(8, dtype=np.float64), sk)

    payload["cases"] = {}
    payload["cases"]["dense_numpy_rotation_key"] = try_case(
        "dense_numpy_rotation_key",
        lambda: {"ciphertext": ct_info(engine.multiply_matrix(matrix, ct, rot_key))},
    )

    plain_holder = {}
    def encode_plain():
        plain = engine.encode_to_plain_matrix(matrix)
        plain_holder["plain"] = plain
        return {"encoded": obj_info(plain)}

    payload["cases"]["encode_plain_matrix"] = try_case(
        "encode_plain_matrix",
        encode_plain,
    )
    if "plain" in plain_holder:
        payload["cases"]["plain_matrix_multiply"] = try_case(
            "plain_matrix_multiply",
            lambda: {"ciphertext": ct_info(engine.multiply_matrix(plain_holder["plain"], ct, mm_key))},
        )

    light_holder = {}
    def encode_light():
        light = engine.encode_to_light_plain_matrix(matrix)
        light_holder["light"] = light
        return {"encoded": obj_info(light)}

    payload["cases"]["encode_light_plain_matrix"] = try_case(
        "encode_light_plain_matrix",
        encode_light,
    )
    if "light" in light_holder:
        payload["cases"]["light_plain_matrix_multiply"] = try_case(
            "light_plain_matrix_multiply",
            lambda: {"ciphertext": ct_info(engine.multiply_matrix(light_holder["light"], ct, mm_key))},
        )

    payload["cases"]["encode_plain_matrix_diagonal_indices"] = try_case(
        "encode_plain_matrix_diagonal_indices",
        lambda: {"encoded": obj_info(engine.encode_to_plain_matrix(matrix, [0, 1]))},
    )
    payload["cases"]["encode_light_plain_matrix_diagonal_indices"] = try_case(
        "encode_light_plain_matrix_diagonal_indices",
        lambda: {"encoded": obj_info(engine.encode_to_light_plain_matrix(matrix, [0, 1]))},
    )

    write_json(RESULTS / "ckks_matrix_api_probe.json", payload)
    print(RESULTS / "ckks_matrix_api_probe.json")


if __name__ == "__main__":
    main()
