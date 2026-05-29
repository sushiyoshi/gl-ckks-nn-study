from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import RESULTS, write_csv


def load_json(name: str) -> dict:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def read_csv_rows(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = []
    obj = load_json("object_sizes.json")
    steady = load_json("steady_state_packed_pca32.json")
    large = load_json("large_batch_packed_pca32.json")
    compare = read_csv_rows("packed_sweep_comparison.csv")

    for r in obj.get("objects", []):
        rows.append({
            "section": "object_sizes",
            "scheme": r.get("scheme"),
            "object_name": r.get("object_name"),
            "supported": r.get("supported"),
            "serialized_nbytes": r.get("serialized_nbytes"),
            "python_object_size": r.get("python_object_size"),
            "notes": r.get("notes"),
        })
    for r in steady.get("results", []):
        rows.append({
            "section": "steady_state",
            "mode": r.get("mode"),
            "n_samples": r.get("n_samples"),
            "semantic_validation_passed": r.get("semantic_validation_passed"),
            "accuracy": r.get("accuracy"),
            "runtime_total_s": r.get("runtime_total_s"),
            "runtime_per_sample_s": r.get("runtime_per_sample_s"),
            "server_only_runtime_per_sample_s": r.get("server_only_runtime_per_sample_s"),
            "offline_total_s": r.get("offline_total_s"),
            "online_total_s": r.get("online_total_s"),
        })
    for r in large.get("results", []):
        rows.append({
            "section": "large_batch",
            "mode": r.get("mode"),
            "n_samples": r.get("n_samples"),
            "semantic_validation_passed": r.get("semantic_validation_passed"),
            "accuracy": r.get("accuracy"),
            "runtime_total_s": r.get("runtime_total_s"),
            "runtime_per_sample_s": r.get("runtime_per_sample_s"),
            "server_only_runtime_per_sample_s": r.get("server_only_runtime_per_sample_s"),
            "skip_reason": r.get("skip_reason"),
        })
    for r in compare:
        rows.append({
            "section": "throughput_compare",
            "condition": r.get("condition"),
            "n_samples": r.get("n_samples"),
            "runtime_per_sample_s": r.get("runtime_per_sample_s"),
            "speedup_vs_ckks_dense": r.get("speedup_vs_ckks_dense"),
            "speedup_vs_gl": r.get("speedup_vs_gl"),
        })
    write_csv(RESULTS / "final_packed_pca32_summary.csv", rows)
    summary_md = [
        "# Packed PCA32 Summary",
        "",
        f"- object size rows: {len(obj.get('objects', []))}",
        f"- steady-state rows: {len(steady.get('results', []))}",
        f"- large-batch rows: {len(large.get('results', []))}",
        f"- throughput compare rows: {len(compare)}",
        "",
        "## Notes",
        "",
        "- GL object serialization is unsupported in this build; object sizes use approximate Python object size only.",
        "- CKKS object sizes use library serialize APIs.",
        "- Dense CKKS large-batch runs may be skipped at high sample counts when runtime is prohibitive.",
    ]
    (RESULTS / "final_packed_pca32_summary.md").write_text("\n".join(summary_md) + "\n")
    print(RESULTS / "final_packed_pca32_summary.csv")


if __name__ == "__main__":
    main()
