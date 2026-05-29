from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.fhe_gl import check_plain_weight_matrix_multiply, flat_to_shape, run_ct_ct_microbenchmark
from src.logging_utils import RESULTS, ct_info, elapsed, exception_record, now_seconds, write_csv, write_json
from src.metrics import accuracy, argmax_agreement, error_metrics
from src.model import load_model


def vector_column(values: np.ndarray, rows: int, shape: tuple[int, ...]) -> np.ndarray:
    matrix = np.zeros((rows, 1), dtype=np.float64)
    matrix[: values.size, 0] = values
    return flat_to_shape(matrix.ravel(), shape)


def bias_column(values: np.ndarray, rows: int, shape: tuple[int, ...]) -> np.ndarray:
    return vector_column(values, rows, shape)


def decrypt_column(engine, sk, ct, n: int) -> np.ndarray:
    dec = np.asarray(engine.decrypt(ct, sk), dtype=np.float64).ravel()
    return dec[:n]


def write_unsupported(reason: str, details: dict) -> None:
    path = RESULTS / "unsupported_gl_plain_weight.md"
    path.write_text(
        "# GL plain-weight inference unsupported\n\n"
        f"Reason: {reason}\n\n"
        "Details:\n\n"
        "```json\n"
        f"{json.dumps(details, indent=2, default=str)}\n"
        "```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--microbenchmark-only", action="store_true")
    args = parser.parse_args()

    from desilofhe import GLEngine

    model, arrays = load_model()
    poly = np.load(Path("data") / f"relu_poly_degree{args.degree}.npz")
    coeffs = poly["coeffs"]
    x = arrays["x_test_scaled"][: args.n_samples]
    y = arrays["y_test"][: args.n_samples]
    plain = model.poly_forward(x, coeffs)["logits"]

    rows = []
    meta = {"n_samples_requested": args.n_samples}
    try:
        engine = GLEngine()
        shape = tuple(getattr(engine, "shape"))
        slots = int(np.prod(shape))
        meta.update({"shape": shape, "slot_count": slots, "packing_utilization_input": 65 / slots, "padding_input": slots - 65})
        t = now_seconds()
        sk = engine.create_secret_key()
        mm_key = engine.create_matrix_multiplication_key(sk)
        had_key = engine.create_hadamard_multiplication_key(sk)
        meta["key_generation_s"] = elapsed(t)
        caps = check_plain_weight_matrix_multiply(engine, sk, mm_key)
        meta["plain_weight_matrix_multiply"] = caps
        supported = caps.get("numpy_16x64_times_ct", {}).get("ok", False)
        if args.microbenchmark_only or not supported:
            mb = run_ct_ct_microbenchmark(engine, sk, mm_key)
            meta["ct_ct_microbenchmark"] = mb
            reason = "GLEngine.matrix_multiply did not accept numpy 16x64 plain weight times GLCiphertext" if not supported else "microbenchmark-only requested"
            write_unsupported(reason, meta)
            rows.append({"ok": False, "unsupported": reason})
        else:
            decoded = []
            w1 = model.w1.T.astype(np.float64)
            w2 = model.w2.T.astype(np.float64)
            for i, sample in enumerate(x):
                try:
                    t = now_seconds()
                    ct = engine.encrypt(vector_column(sample, 64, shape), sk)
                    enc_s = elapsed(t)
                    t = now_seconds()
                    ct = engine.matrix_multiply(w1, ct, mm_key)
                    ct = engine.add(ct, bias_column(model.b1, 16, shape))
                    linear1_s = elapsed(t)
                    level_l1 = getattr(ct, "level", None)
                    t = now_seconds()
                    ct = engine.evaluate_polynomial(ct, coeffs.astype(np.float64), had_key)
                    activation_s = elapsed(t)
                    level_act = getattr(ct, "level", None)
                    t = now_seconds()
                    ct = engine.matrix_multiply(w2, ct, mm_key)
                    ct = engine.add(ct, bias_column(model.b2, 10, shape))
                    linear2_s = elapsed(t)
                    info = ct_info(ct)
                    t = now_seconds()
                    logits = decrypt_column(engine, sk, ct, 10)
                    decrypt_s = elapsed(t)
                    decoded.append(logits)
                    rows.append({
                        "sample_index": i,
                        "true_label": int(y[i]),
                        "plain_poly_pred": int(np.argmax(plain[i])),
                        "fhe_pred": int(np.argmax(logits)),
                        "ok": True,
                        "encrypt_s": enc_s,
                        "linear1_s": linear1_s,
                        "activation_s": activation_s,
                        "linear2_s": linear2_s,
                        "decrypt_s": decrypt_s,
                        "linear1_level": level_l1,
                        "activation_level": level_act,
                        "linear2_level": info.get("level"),
                    })
                except Exception as exc:
                    rec = exception_record(exc, sample_index=i, api="GL plaintext-weight polynomial MLP", ciphertext_level=getattr(locals().get("ct", None), "level", None))
                    rows.append(rec)
                    write_unsupported("GL plaintext-weight MLP path failed during execution", rec)
                    break
            if decoded:
                decoded_arr = np.vstack(decoded)
                err = error_metrics(plain[: len(decoded)], decoded_arr, "fhe_numeric_logits")
                meta.update({
                    "accuracy_fhe_decrypted": accuracy(decoded_arr, y[: len(decoded)]),
                    "argmax_agreement": argmax_agreement(plain[: len(decoded)], decoded_arr),
                    **err,
                })
                if err["fhe_numeric_logits_relative_l2"] > 1e-2:
                    meta["semantic_validation_supported"] = False
                    write_unsupported(
                        "GLEngine.matrix_multiply accepts plaintext weights, but this padded vector-to-GL-matrix mapping did not reproduce the plaintext polynomial MLP within tolerance.",
                        meta,
                    )
                else:
                    meta["semantic_validation_supported"] = True
    except Exception as exc:
        rows.append(exception_record(exc, api="GL setup"))
        write_unsupported("GL setup failed", rows[-1])

    write_json(RESULTS / "gl_results.json", meta)
    write_csv(RESULTS / "gl_results.csv", rows)
    print(RESULTS / "gl_results.csv")


if __name__ == "__main__":
    main()
