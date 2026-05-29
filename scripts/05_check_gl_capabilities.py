from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.fhe_gl import check_plain_weight_matrix_multiply, run_ct_ct_microbenchmark
from src.logging_utils import RESULTS, ct_info, elapsed, exception_record, now_seconds, write_json


def main() -> None:
    import desilofhe
    from desilofhe import Engine, GLEngine

    payload = {"desilofhe_version_attr": getattr(desilofhe, "__version__", None)}
    try:
        e = Engine()
        t = now_seconds()
        sk = e.create_secret_key()
        payload["ckks"] = {
            "engine": True,
            "slot_count": getattr(e, "slot_count", None),
            "max_level": getattr(e, "max_level", None),
            "secret_key_generation_s": elapsed(t),
            "has_encrypt": hasattr(e, "encrypt"),
            "has_decrypt": hasattr(e, "decrypt"),
            "has_multiply_matrix": hasattr(e, "multiply_matrix"),
            "has_evaluate_polynomial": hasattr(e, "evaluate_polynomial"),
            "has_evaluate_chebyshev_polynomial": hasattr(e, "evaluate_chebyshev_polynomial"),
        }
    except Exception as exc:
        payload["ckks"] = exception_record(exc)

    try:
        g = GLEngine()
        t = now_seconds()
        gsk = g.create_secret_key()
        mm_key = g.create_matrix_multiplication_key(gsk)
        payload["gl"] = {
            "engine": True,
            "shape": getattr(g, "shape", None),
            "slot_count": getattr(g, "slot_count", None),
            "max_level": getattr(g, "max_level", None),
            "key_generation_s": elapsed(t),
            "has_encrypt": hasattr(g, "encrypt"),
            "has_decrypt": hasattr(g, "decrypt"),
            "has_matrix_multiply": hasattr(g, "matrix_multiply"),
            "has_evaluate_polynomial": hasattr(g, "evaluate_polynomial"),
            "plain_weight_matrix_multiply": check_plain_weight_matrix_multiply(g, gsk, mm_key),
        }
        try:
            payload["gl"]["ct_ct_microbenchmark"] = run_ct_ct_microbenchmark(g, gsk, mm_key)
        except Exception as exc:
            payload["gl"]["ct_ct_microbenchmark"] = exception_record(exc)
    except Exception as exc:
        payload["gl"] = exception_record(exc)

    write_json(RESULTS / "gl_capabilities.json", payload)
    print(RESULTS / "gl_capabilities.json")


if __name__ == "__main__":
    main()
