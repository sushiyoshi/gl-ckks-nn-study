from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .logging_utils import DATA, SEED
from .model import load_digits_split
from .model_pca32 import expand_eval_split
from .polynomial import eval_power_polynomial, relu


@dataclass
class Raw64MLP:
    scaler: StandardScaler
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    metadata: dict[str, Any]

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.transform(x)

    def relu_forward(self, x64: np.ndarray) -> dict[str, np.ndarray]:
        z1 = x64 @ self.w1 + self.b1
        h1 = relu(z1)
        logits = h1 @ self.w2 + self.b2
        return {"z1": z1, "h1": h1, "logits": logits}

    def poly_forward(self, x64: np.ndarray, coeffs: np.ndarray) -> dict[str, np.ndarray]:
        z1 = x64 @ self.w1 + self.b1
        h1 = eval_power_polynomial(z1, coeffs)
        logits = h1 @ self.w2 + self.b2
        return {"z1": z1, "h1": h1, "logits": logits}


def train_raw64_model(
    hidden: int = 32,
    max_iter: int = 800,
    test_eval_size: int | None = None,
) -> tuple[Raw64MLP, dict[str, np.ndarray]]:
    x_train, x_test_base, y_train, y_test_base = load_digits_split()
    scaler = StandardScaler()
    x_train_64 = scaler.fit_transform(x_train)
    x_test_64_base = scaler.transform(x_test_base)
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        random_state=SEED,
        early_stopping=True,
        n_iter_no_change=30,
    )
    clf.fit(x_train_64, y_train)
    model = Raw64MLP(
        scaler=scaler,
        w1=clf.coefs_[0].astype(np.float64),
        b1=clf.intercepts_[0].astype(np.float64),
        w2=clf.coefs_[1].astype(np.float64),
        b2=clf.intercepts_[1].astype(np.float64),
        metadata={
            "hidden": hidden,
            "feature_dim": 64,
            "classes": clf.classes_.tolist(),
            "n_iter": int(clf.n_iter_),
            "loss": float(clf.loss_),
            "seed": SEED,
            "split_policy": "sklearn digits, train_test_split(test_size=0.25, random_state=42, stratify=y)",
        },
    )
    if test_eval_size is None:
        x_test_64 = x_test_64_base
        y_test = y_test_base
    else:
        x_test_64, y_test = expand_eval_split(x_test_64_base, y_test_base, target_size=test_eval_size)
    arrays = {
        "x_train": x_train,
        "x_test_base": x_test_base,
        "y_train": y_train,
        "y_test_base": y_test_base,
        "x_train_64": x_train_64,
        "x_test_64": x_test_64,
        "x_test_64_base": x_test_64_base,
        "y_test": y_test,
    }
    return model, arrays


def save_raw64_model(model: Raw64MLP, arrays: dict[str, np.ndarray], path: Path = DATA / "raw64_mlp.joblib") -> None:
    path.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "arrays": arrays}, path)


def load_raw64_model(path: Path = DATA / "raw64_mlp.joblib") -> tuple[Raw64MLP, dict[str, np.ndarray]]:
    payload = joblib.load(path)
    return payload["model"], payload["arrays"]
