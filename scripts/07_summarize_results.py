from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, write_csv


def read_json_if_exists(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def read_csv_first(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def main() -> None:
    plain = read_csv_first(RESULTS / "plain_baseline.csv")
    ckks = read_json_if_exists(RESULTS / "ckks_results.json")
    gl = read_json_if_exists(RESULTS / "gl_results.json")
    rows = [
        {
            "condition": "plain_relu",
            "accuracy": plain.get("accuracy_original_relu"),
            "notes": "scikit-learn MLPClassifier ReLU baseline",
        },
        {
            "condition": "plain_polynomial_relu",
            "accuracy": plain.get("accuracy_poly_plain"),
            "argmax_agreement": plain.get("argmax_agreement_relu_poly"),
            "notes": "same trained weights, ReLU replaced by fitted power polynomial",
        },
        {
            "condition": "ckks_polynomial_relu",
            "accuracy": ckks.get("accuracy_fhe_decrypted"),
            "argmax_agreement": ckks.get("argmax_agreement"),
            "slot_count": ckks.get("slot_count"),
            "packing_utilization_input": ckks.get("packing_utilization_input"),
            "notes": "input encrypted, weights plaintext, CKKS multiply_matrix",
        },
        {
            "condition": "gl_polynomial_relu",
            "accuracy": gl.get("accuracy_fhe_decrypted"),
            "argmax_agreement": gl.get("argmax_agreement"),
            "slot_count": gl.get("slot_count"),
            "packing_utilization_input": gl.get("packing_utilization_input"),
            "notes": "input encrypted, weights plaintext if public API path succeeded; see unsupported markdown otherwise",
        },
    ]
    write_csv(RESULTS / "summary.csv", rows)
    print(RESULTS / "summary.csv")


if __name__ == "__main__":
    main()
