from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, write_json
from src.metrics import accuracy
from src.model_pca32 import save_pca32_model, train_pca32_model


def main() -> None:
    model, arrays = train_pca32_model(test_eval_size=8192)
    save_pca32_model(model, arrays)
    train_logits = model.relu_forward(arrays["x_train_32"])["logits"]
    test_logits = model.relu_forward(arrays["x_test_32"])["logits"]
    write_json(RESULTS / "pca32_train.json", {
        **model.metadata,
        "accuracy_train_relu": accuracy(train_logits, arrays["y_train"]),
        "accuracy_test_relu": accuracy(test_logits, arrays["y_test"]),
    })
    print(RESULTS / "pca32_train.json")


if __name__ == "__main__":
    main()
