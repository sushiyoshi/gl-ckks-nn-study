from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gl_encrypted_weight import mutation_results
from src.logging_utils import RESULTS, write_csv, write_json


def flatten_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "mutation": case.get("mutation"),
        "ok": case.get("ok"),
        "n_samples": case.get("n_samples"),
        "degree": case.get("degree"),
        "semantic_validation_passed_against_correct": case.get("semantic_validation_passed_against_correct"),
        "semantic_validation_passed_against_mutated": case.get("semantic_validation_passed_against_mutated"),
        "correct_relative_l2": case.get("correct_relative_l2"),
        "mutated_relative_l2": case.get("mutated_relative_l2"),
        "argmax_agreement_against_correct": case.get("argmax_agreement_against_correct"),
        "argmax_agreement_against_mutated": case.get("argmax_agreement_against_mutated"),
        "ct_W1_type": case.get("ct_W1_type"),
        "ct_W2_type": case.get("ct_W2_type"),
        "ct_X_type": case.get("ct_X_type"),
        "matrix_multiply_operand_kinds": case.get("matrix_multiply_operand_kinds"),
        "weight_encryption_s": case.get("weight_encryption_s"),
        "input_encryption_s": case.get("input_encryption_s"),
        "server_only_s": case.get("server_only_s"),
        "total_s": case.get("total_s"),
        "runtime_total_s": case.get("runtime_total_s"),
        "runtime_per_sample_s": case.get("runtime_per_sample_s"),
        "failure_category": case.get("failure_category"),
        "exception": case.get("exception"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=32)
    args = parser.parse_args()

    cases = []
    for mutation in [
        "correct_weight",
        "zero_W1",
        "zero_W2",
        "random_W1",
        "shuffled_W1_batches",
        "wrong_key_or_wrong_weight",
    ]:
        try:
            case = mutation_results(n_samples=args.n_samples, degree=args.degree, mutation=mutation)
        except Exception as exc:
            case = {
                "ok": False,
                "mutation": mutation,
                "n_samples": args.n_samples,
                "degree": args.degree,
                "semantic_validation_passed_against_correct": False,
                "semantic_validation_passed_against_mutated": False,
                "correct_relative_l2": None,
                "mutated_relative_l2": None,
                "argmax_agreement_against_correct": None,
                "argmax_agreement_against_mutated": None,
                "ct_W1_type": None,
                "ct_W2_type": None,
                "ct_X_type": None,
                "matrix_multiply_operand_kinds": None,
                "weight_encryption_s": None,
                "input_encryption_s": None,
                "server_only_s": None,
                "total_s": None,
                "runtime_total_s": None,
                "runtime_per_sample_s": None,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "failure_category": "script_failure",
            }
        cases.append(case)

    payload = {
        "ok": all(case.get("ok") for case in cases),
        "degree": args.degree,
        "n_samples": args.n_samples,
        "cases": cases,
        "mutation_summary": {
            "correct_weight_passed": any(
                c.get("mutation") == "correct_weight" and c.get("semantic_validation_passed_against_correct") for c in cases
            ),
            "mutations_failed_against_correct": [
                c.get("mutation")
                for c in cases
                if c.get("mutation") != "correct_weight" and not c.get("semantic_validation_passed_against_correct")
            ],
        },
    }

    rows = [flatten_case(case) for case in cases]
    write_json(RESULTS / "gl_encrypted_weight_mutation_tests.json", payload)
    write_csv(RESULTS / "gl_encrypted_weight_mutation_tests.csv", rows)
    print(RESULTS / "gl_encrypted_weight_mutation_tests.json")


if __name__ == "__main__":
    main()
