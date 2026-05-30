from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_block_mlp import raw64_polynomial
from src.logging_utils import RESULTS, write_json
from src.metrics import accuracy, argmax_agreement
from src.model_raw64 import save_raw64_model, train_raw64_model


def main() -> None:
    model, arrays = train_raw64_model(test_eval_size=8192)
    save_raw64_model(model, arrays)
    radius, coeffs = raw64_polynomial()
    train_relu = model.relu_forward(arrays["x_train_64"])["logits"]
    test_relu = model.relu_forward(arrays["x_test_64"])["logits"]
    train_poly = model.poly_forward(arrays["x_train_64"], coeffs)["logits"]
    test_poly = model.poly_forward(arrays["x_test_64"], coeffs)["logits"]
    write_json(
        RESULTS / "raw64_train.json",
        {
            **model.metadata,
            "ok": True,
            "task_type": "raw64_mlp_poly_relu",
            "model_path": "data/raw64_mlp.joblib",
            "logical_dims": [64, 32, 10],
            "test_eval_size": int(arrays["x_test_64"].shape[0]),
            "test_base_size": int(arrays["x_test_64_base"].shape[0]),
            "eval_resampling": "deterministic replacement resampling from x_test_64_base/y_test_base for throughput evaluation; not unique 8192-sample accuracy.",
            "polynomial_degree": 3,
            "polynomial_radius": radius,
            "coefficients": coeffs.tolist(),
            "coefficients_power_basis_order": "ascending",
            "accuracy_train_relu": accuracy(train_relu, arrays["y_train"]),
            "accuracy_test_relu": accuracy(test_relu, arrays["y_test"]),
            "accuracy_train_poly": accuracy(train_poly, arrays["y_train"]),
            "accuracy_test_poly": accuracy(test_poly, arrays["y_test"]),
            "argmax_agreement_train_relu_vs_poly": argmax_agreement(train_relu, train_poly),
            "argmax_agreement_test_relu_vs_poly": argmax_agreement(test_relu, test_poly),
        },
    )
    print(RESULTS / "raw64_train.json")


if __name__ == "__main__":
    main()
