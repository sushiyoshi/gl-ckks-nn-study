from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, write_json
from src.metrics import accuracy
from src.model import save_model, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=16)
    args = parser.parse_args()
    model, arrays = train_model(hidden=args.hidden)
    save_model(model, arrays)
    train_logits = model.relu_forward(arrays["x_train_scaled"])["logits"]
    test_logits = model.relu_forward(arrays["x_test_scaled"])["logits"]
    write_json(RESULTS / "train_plain_mlp.json", {
        **model.metadata,
        "accuracy_train_relu": accuracy(train_logits, arrays["y_train"]),
        "accuracy_test_relu": accuracy(test_logits, arrays["y_test"]),
    })
    print(RESULTS / "train_plain_mlp.json")


if __name__ == "__main__":
    main()
