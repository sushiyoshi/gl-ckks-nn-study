from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import DATA, RESULTS, write_csv
from src.metrics import accuracy, argmax_agreement, error_metrics
from src.model import load_model
from src.polynomial import out_of_interval_fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    args = parser.parse_args()
    model, arrays = load_model()
    poly = np.load(DATA / f"relu_poly_degree{args.degree}.npz")
    coeffs = poly["coeffs"]
    interval = tuple(float(x) for x in poly["interval"])
    y = arrays["y_test"]
    x = arrays["x_test_scaled"]
    relu_out = model.relu_forward(x)
    poly_out = model.poly_forward(x, coeffs)
    clipped_out = model.poly_forward(x, coeffs, clip_interval=interval)
    row = {
        "degree": args.degree,
        "interval_low": interval[0],
        "interval_high": interval[1],
        "accuracy_original_relu": accuracy(relu_out["logits"], y),
        "accuracy_poly_plain": accuracy(poly_out["logits"], y),
        "accuracy_poly_plain_clipped": accuracy(clipped_out["logits"], y),
        "argmax_agreement_relu_poly": argmax_agreement(relu_out["logits"], poly_out["logits"]),
        "hidden_pre_activation_out_of_interval": out_of_interval_fraction(relu_out["z1"], interval),
        **error_metrics(relu_out["h1"], poly_out["h1"], "hidden_relu_vs_poly"),
        **error_metrics(relu_out["logits"], poly_out["logits"], "logits_relu_vs_poly"),
        **error_metrics(poly_out["logits"], clipped_out["logits"], "logits_poly_vs_clipped"),
    }
    write_csv(RESULTS / "plain_baseline.csv", [row])
    print(RESULTS / "plain_baseline.csv")


if __name__ == "__main__":
    main()
