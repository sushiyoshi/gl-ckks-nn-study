from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - handled as unsupported at runtime
    np = None  # type: ignore[assignment]

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, elapsed, now_seconds, write_csv, write_json


def safe_array(value: Any) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy is unavailable")
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        arr = np.real_if_close(arr, tol=1e5)
    return np.asarray(arr, dtype=np.float64)


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | bool]:
    if np is None:
        raise RuntimeError("numpy is unavailable")
    diff = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    denom = float(np.linalg.norm(reference.ravel()) or 1.0)
    rel_l2 = float(np.linalg.norm(diff.ravel()) / denom)
    linf = float(np.max(np.abs(diff)))
    mae = float(np.mean(np.abs(diff)))
    allclose = bool(np.allclose(actual, reference, rtol=1e-3, atol=1e-3))
    return {
        "relative_l2": rel_l2,
        "linf": linf,
        "mae": mae,
        "allclose": allclose,
    }


def classify_failure(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "module" in text and "openfhe" in text:
        return "import_error"
    if "wheel" in text or "build" in text or "cmake" in text or "dependency" in text:
        return "build_dependency_missing"
    if "unsupported" in text:
        return "unsupported"
    return "api_error"


def make_context(d: int):
    from openfhe import (
        CCParamsCKKSRNS,
        GenCryptoContext,
        PKESchemeFeature,
        ScalingTechnique,
        SecretKeyDist,
    )

    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(7)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    params.SetSecretKeyDist(SecretKeyDist.UNIFORM_TERNARY)
    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)
    keys = cc.KeyGen()
    if not keys:
        raise RuntimeError("KeyGen returned no keys")
    t = now_seconds()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)
    keygen_s = elapsed(t)
    return cc, keys, keygen_s


def encrypt_matrix(onp: Any, cc: Any, public_key: Any, matrix: np.ndarray, batch_size: int):
    return onp.array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=public_key,
    )


def run_case(d: int) -> dict[str, Any]:
    t_total = now_seconds()
    if np is None:
        return {
            "ok": False,
            "d": d,
            "semantic_validation_passed": False,
            "failure_category": "build_dependency_missing",
            "unsupported_reason": "numpy_missing",
            "exception_type": "ModuleNotFoundError",
            "exception": "No module named 'numpy'",
            "runtime_total_s": elapsed(t_total),
            "notes": "numpy is required for OpenFHE-NumPy experiments",
        }
    try:
        import openfhe_numpy as onp
    except Exception as exc:
        return {
            "ok": False,
            "d": d,
            "semantic_validation_passed": False,
            "failure_category": "import_error",
            "unsupported_reason": classify_failure(exc),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "runtime_total_s": elapsed(t_total),
        }

    rng = np.random.default_rng(42 + d)
    A = rng.normal(0.0, 0.25, size=(d, d)).astype(np.float64)
    B = rng.normal(0.0, 0.25, size=(d, d)).astype(np.float64)
    expected = A @ B

    try:
        t = now_seconds()
        cc, keys, keygen_s = make_context(d)
        context_s = elapsed(t)

        batch_size = d
        t = now_seconds()
        tensor_A = encrypt_matrix(onp, cc, keys.publicKey, A, batch_size)
        tensor_B = encrypt_matrix(onp, cc, keys.publicKey, B, batch_size)
        encryption_s = elapsed(t)

        rotation_keygen_s = None
        try:
            t = now_seconds()
            onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, getattr(tensor_A, "ncols", d))
            rotation_keygen_s = elapsed(t)
        except Exception as exc:
            rotation_keygen_s = None
            rotation_keygen_error = {
                "rotation_keygen_error_type": type(exc).__name__,
                "rotation_keygen_error": str(exc),
            }
        else:
            rotation_keygen_error = {}

        t = now_seconds()
        tensor_C = tensor_A @ tensor_B
        matmul_s = elapsed(t)

        t = now_seconds()
        decrypted = tensor_C.decrypt(keys.secretKey, unpack_type="original")
        decryption_s = elapsed(t)

        actual = safe_array(decrypted)
        if actual.shape != expected.shape:
            actual = np.reshape(actual, expected.shape)

        metric_values = metrics(expected, actual)
        semantic_validation_passed = bool(metric_values["relative_l2"] < 1e-3 and metric_values["allclose"])
        payload = {
            "ok": True,
            "d": d,
            "scheme_or_backend": "CKKS",
            "library": "OpenFHE-NumPy",
            "encrypted_operands": "ct_ct",
            "packing_layout": "single-ciphertext row-major matrix",
            "batch_size": batch_size,
            "tensor_A_shape": list(A.shape),
            "tensor_B_shape": list(B.shape),
            "tensor_C_shape": list(actual.shape),
            "ring_dimension": getattr(cc, "GetRingDimension", lambda: None)(),
            "keygen_s": keygen_s,
            "rotation_or_mm_keygen_s": rotation_keygen_s,
            "encryption_s": encryption_s,
            "matmul_s": matmul_s,
            "decryption_s": decryption_s,
            "context_setup_s": context_s,
            "runtime_total_s": elapsed(t_total),
            "semantic_validation_passed": semantic_validation_passed,
            "relative_l2": metric_values["relative_l2"],
            "linf": metric_values["linf"],
            "mae": metric_values["mae"],
            "allclose": metric_values["allclose"],
            "expected_trace": float(np.trace(expected)),
            "actual_trace": float(np.trace(actual)),
            "notes": "OpenFHE-NumPy single-ciphertext baseline; rotation keygen is best-effort",
            **rotation_keygen_error,
        }
        return payload
    except Exception as exc:
        return {
            "ok": False,
            "d": d,
            "scheme_or_backend": "CKKS",
            "library": "OpenFHE-NumPy",
            "encrypted_operands": "ct_ct",
            "packing_layout": "single-ciphertext row-major matrix",
            "semantic_validation_passed": False,
            "failure_category": classify_failure(exc),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "runtime_total_s": elapsed(t_total),
            "notes": "OpenFHE-NumPy toy matmul failed before semantic validation",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=2)
    args = parser.parse_args()

    payload = run_case(args.d)
    rows = [payload]
    write_json(RESULTS / "openfhe_numpy_ckks_matmul_toy.json", payload)
    write_csv(RESULTS / "openfhe_numpy_ckks_matmul_toy.csv", rows)
    print(RESULTS / "openfhe_numpy_ckks_matmul_toy.json")


if __name__ == "__main__":
    main()
