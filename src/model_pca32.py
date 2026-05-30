from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .logging_utils import DATA, SEED
from .model import load_digits_split
from .polynomial import eval_power_polynomial, relu


@dataclass
class PCA32MLP:
    scaler: StandardScaler
    pca: PCA
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    metadata: dict[str, Any]

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(x))

    def relu_forward(self, x32: np.ndarray) -> dict[str, np.ndarray]:
        z1 = x32 @ self.w1 + self.b1
        h1 = relu(z1)
        logits = h1 @ self.w2 + self.b2
        return {"z1": z1, "h1": h1, "logits": logits}

    def poly_forward(self, x32: np.ndarray, coeffs: np.ndarray) -> dict[str, np.ndarray]:
        z1 = x32 @ self.w1 + self.b1
        h1 = eval_power_polynomial(z1, coeffs)
        logits = h1 @ self.w2 + self.b2
        return {"z1": z1, "h1": h1, "logits": logits}


def expand_eval_split(
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    target_size: int,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    if target_size < 0:
        raise ValueError(f"target_size must be non-negative, got {target_size}")
    if target_size <= len(x_test):
        return x_test[:target_size], y_test[:target_size]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_test), size=target_size, replace=True)
    return x_test[idx], y_test[idx]


def require_sample_count(n_samples: int, available: int, *, label: str) -> None:
    if n_samples < 0:
        raise ValueError(f"{label} must be non-negative, got {n_samples}")
    if n_samples > available:
        raise ValueError(f"requested {n_samples} samples for {label}, but only {available} are available")


def train_pca32_model(
    hidden: int = 32,
    max_iter: int = 800,
    test_eval_size: int | None = None,
) -> tuple[PCA32MLP, dict[str, np.ndarray]]:
    x_train, x_test, y_train, y_test = load_digits_split()
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    pca = PCA(n_components=32, random_state=SEED)
    x_train_32 = pca.fit_transform(x_train_s)
    x_test_32_base = pca.transform(x_test_s)
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
    clf.fit(x_train_32, y_train)
    model = PCA32MLP(
        scaler=scaler,
        pca=pca,
        w1=clf.coefs_[0].astype(np.float64),
        b1=clf.intercepts_[0].astype(np.float64),
        w2=clf.coefs_[1].astype(np.float64),
        b2=clf.intercepts_[1].astype(np.float64),
        metadata={
            "hidden": hidden,
            "pca_components": 32,
            "classes": clf.classes_.tolist(),
            "n_iter": int(clf.n_iter_),
            "loss": float(clf.loss_),
            "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "seed": SEED,
        },
    )
    if test_eval_size is None:
        x_test_32 = x_test_32_base
        y_test_eval = y_test
    else:
        x_test_32, y_test_eval = expand_eval_split(x_test_32_base, y_test, target_size=test_eval_size)
    arrays = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test_eval,
        "x_train_32": x_train_32,
        "x_test_32": x_test_32,
        "x_test_32_base": x_test_32_base,
        "y_test_base": y_test,
    }
    return model, arrays


def save_pca32_model(model: PCA32MLP, arrays: dict[str, np.ndarray], path: Path = DATA / "pca32_mlp.joblib") -> None:
    path.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "arrays": arrays}, path)


def load_pca32_model(path: Path = DATA / "pca32_mlp.joblib") -> tuple[PCA32MLP, dict[str, np.ndarray]]:
    payload = joblib.load(path)
    return payload["model"], payload["arrays"]
