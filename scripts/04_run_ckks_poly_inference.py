from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.fhe_ckks import run_one
from src.logging_utils import RESULTS, elapsed, exception_record, now_seconds, write_csv, write_json
from src.metrics import accuracy, argmax_agreement, error_metrics
from src.model import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=10)
    args = parser.parse_args()

    import desilofhe
    from desilofhe import Engine

    model, arrays = load_model()
    poly = np.load(Path("data") / f"relu_poly_degree{args.degree}.npz")
    coeffs = poly["coeffs"]
    x = arrays["x_test_scaled"][: args.n_samples]
    y = arrays["y_test"][: args.n_samples]
    plain = model.poly_forward(x, coeffs)["logits"]

    rows = []
    meta = {"desilofhe_version_attr": getattr(desilofhe, "__version__", None), "n_samples_requested": args.n_samples}
    try:
        engine = Engine()
        slots = int(getattr(engine, "slot_count", 8192))
        meta["slot_count"] = slots
        meta["packing_utilization_input"] = 65 / slots
        meta["padding_input"] = slots - 65
        t = now_seconds()
        sk = engine.create_secret_key()
        rot_key = engine.create_rotation_key(sk)
        relin_key = engine.create_relinearization_key(sk)
        meta["key_generation_s"] = elapsed(t)
        decoded = []
        for i, sample in enumerate(x):
            try:
                out = run_one(engine, sk, rot_key, relin_key, sample, model.w1, model.b1, model.w2, model.b2, coeffs)
                decoded.append(out["logits"])
                row = {
                    "sample_index": i,
                    "true_label": int(y[i]),
                    "plain_poly_pred": int(np.argmax(plain[i])),
                    "fhe_pred": int(np.argmax(out["logits"])),
                    "ok": True,
                }
                for stage, info in out["log"].items():
                    if isinstance(info, dict):
                        for k, v in info.items():
                            row[f"{stage}_{k}"] = v
                    else:
                        row[stage] = info
                rows.append(row)
            except Exception as exc:
                rows.append(exception_record(exc, sample_index=i, api="CKKS polynomial MLP"))
                break
        if decoded:
            decoded_arr = np.vstack(decoded)
            meta.update({
                "accuracy_fhe_decrypted": accuracy(decoded_arr, y[: len(decoded)]),
                "argmax_agreement": argmax_agreement(plain[: len(decoded)], decoded_arr),
                **error_metrics(plain[: len(decoded)], decoded_arr, "fhe_numeric_logits"),
            })
    except Exception as exc:
        rows.append(exception_record(exc, api="CKKS setup"))

    write_json(RESULTS / "ckks_results.json", meta)
    write_csv(RESULTS / "ckks_results.csv", rows)
    print(RESULTS / "ckks_results.csv")


if __name__ == "__main__":
    main()
