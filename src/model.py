from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .logging_utils import DATA, SEED
from .polynomial import eval_power_polynomial, relu


@dataclass
class PlainMLP:
    scaler: StandardScaler
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    metadata: dict[str, Any]

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.transform(x)

    def relu_forward(self, x_scaled: np.ndarray) -> dict[str, np.ndarray]:
        z1 = x_scaled @ self.w1 + self.b1
        h1 = relu(z1)
        logits = h1 @ self.w2 + self.b2
        return {"z1": z1, "h1": h1, "logits": logits}

    def poly_forward(self, x_scaled: np.ndarray, coeffs: np.ndarray, clip_interval: tuple[float, float] | None = None) -> dict[str, np.ndarray]:
        z1 = x_scaled @ self.w1 + self.b1
        poly_input = np.clip(z1, *clip_interval) if clip_interval else z1
        h1 = eval_power_polynomial(poly_input, coeffs)
        logits = h1 @ self.w2 + self.b2
        return {"z1": z1, "h1": h1, "logits": logits}


def load_digits_split(test_size: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    digits = load_digits()
    x = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(np.int64)
    return train_test_split(x, y, test_size=test_size, random_state=SEED, stratify=y)


def train_model(hidden: int = 16, max_iter: int = 800) -> tuple[PlainMLP, dict[str, np.ndarray]]:
    x_train, x_test, y_train, y_test = load_digits_split()
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
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
    clf.fit(x_train_s, y_train)
    model = PlainMLP(
        scaler=scaler,
        w1=clf.coefs_[0].astype(np.float64),
        b1=clf.intercepts_[0].astype(np.float64),
        w2=clf.coefs_[1].astype(np.float64),
        b2=clf.intercepts_[1].astype(np.float64),
        metadata={
            "hidden": hidden,
            "classes": clf.classes_.tolist(),
            "n_iter": int(clf.n_iter_),
            "loss": float(clf.loss_),
            "seed": SEED,
        },
    )
    arrays = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test, "x_train_scaled": x_train_s, "x_test_scaled": x_test_s}
    return model, arrays


def save_model(model: PlainMLP, arrays: dict[str, np.ndarray], path: Path = DATA / "plain_mlp.joblib") -> None:
    path.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "arrays": arrays}, path)


def load_model(path: Path = DATA / "plain_mlp.joblib") -> tuple[PlainMLP, dict[str, np.ndarray]]:
    payload = joblib.load(path)
    return payload["model"], payload["arrays"]
