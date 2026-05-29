from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import DATA, RESULTS, write_json
from src.model import load_model
from src.polynomial import choose_interval, fit_relu_power_polynomial, out_of_interval_fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--interval", nargs=2, type=float)
    args = parser.parse_args()
    model, arrays = load_model()
    z1 = model.relu_forward(arrays["x_train_scaled"])["z1"]
    interval = tuple(args.interval) if args.interval else choose_interval(z1)
    coeffs = fit_relu_power_polynomial(args.degree, interval)
    path = DATA / f"relu_poly_degree{args.degree}.npz"
    np.savez(path, coeffs=coeffs, interval=np.asarray(interval), degree=args.degree)
    write_json(RESULTS / "relu_polynomial.json", {
        "degree": args.degree,
        "interval": interval,
        "coefficients_power_basis_ascending": coeffs.tolist(),
        "train_hidden_pre_activation_out_of_interval": out_of_interval_fraction(z1, interval),
    })
    print(path)


if __name__ == "__main__":
    main()
