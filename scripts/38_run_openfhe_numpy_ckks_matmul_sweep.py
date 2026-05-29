from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR.parents[0]))

from __openfhe_numpy_shared import run_openfhe_numpy_case  # type: ignore
from src.logging_utils import RESULTS, write_csv, write_json


def main() -> None:
    ds = [2, 4, 8, 16, 32]
    rows = [run_openfhe_numpy_case(d) for d in ds]
    payload = {
        "ok": any(bool(row.get("ok")) for row in rows),
        "cases": rows,
        "semantic_validation_passed": all(bool(row.get("semantic_validation_passed")) for row in rows if row.get("ok")),
        "requested_ds": ds,
    }
    write_csv(RESULTS / "openfhe_numpy_ckks_matmul_sweep.csv", rows)
    write_json(RESULTS / "openfhe_numpy_ckks_matmul_sweep.json", payload)
    print(RESULTS / "openfhe_numpy_ckks_matmul_sweep.csv")


if __name__ == "__main__":
    main()
