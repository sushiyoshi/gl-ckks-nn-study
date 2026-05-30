#!/usr/bin/env bash
set -euo pipefail

OUT="results/report_inputs"
mkdir -p "$OUT/key_results" "$OUT/source_snippets"

echo "[collect] writing to $OUT"

{
  echo "===== DATE ====="
  date

  echo
  echo "===== PWD / HOST ====="
  pwd
  hostname || true

  echo
  echo "===== OS ====="
  uname -a
  if command -v sw_vers >/dev/null 2>&1; then
    sw_vers
  fi
  if command -v lsb_release >/dev/null 2>&1; then
    lsb_release -a
  fi

  echo
  echo "===== CPU ====="
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n machdep.cpu.brand_string 2>/dev/null || true
    sysctl -n hw.ncpu 2>/dev/null || true
    sysctl -n hw.memsize 2>/dev/null || true
  fi
  if command -v lscpu >/dev/null 2>&1; then
    lscpu
  fi

  echo
  echo "===== MEMORY ====="
  if command -v vm_stat >/dev/null 2>&1; then
    vm_stat
  fi
  if command -v free >/dev/null 2>&1; then
    free -h
  fi

  echo
  echo "===== DISK ====="
  df -h .

  echo
  echo "===== PYTHON ====="
  which python3 || true
  python3 --version || true
  python3 -m pip --version || true

  echo
  echo "===== LOCAL DESILOFHE PYTHON ====="
  if [ -x "./bin/python3" ]; then
    ./bin/python3 --version
    ./bin/python3 - <<'PY'
import sys
print("executable:", sys.executable)
for name in ["desilofhe", "numpy", "pandas", "sklearn", "joblib"]:
    try:
        mod = __import__(name)
        print(name, "OK", getattr(mod, "__version__", "no_version"))
    except Exception as e:
        print(name, "FAIL", repr(e))
PY
  else
    echo "./bin/python3 not found"
  fi
} | tee "$OUT/environment.txt"

{
  echo "===== GIT REV ====="
  git rev-parse HEAD || true
  git branch --show-current || true

  echo
  echo "===== GIT STATUS ====="
  git status --short --untracked-files=all || true

  echo
  echo "===== GIT DIFF STAT ====="
  git diff --stat || true

  echo
  echo "===== RECENT COMMITS ====="
  git log --oneline -n 10 || true
} | tee "$OUT/git_state.txt"

find results -maxdepth 3 -type f | sort > "$OUT/result_files.txt"

# Copy important result artifacts if present.
for f in \
  results/pca32_train.json \
  results/gl_encrypted_weight_pca32_linear_n450.json \
  results/gl_encrypted_weight_pca32_linear_n8192.json \
  results/gl_encrypted_weight_pca32_two_linear_n450.json \
  results/gl_encrypted_weight_pca32_mlp_n450.json \
  results/gl_encrypted_weight_pca32_mlp_n8192.json \
  results/gl_ctct_matrix_multiply_probe.json \
  results/gl_encrypted_weight_mutation_tests.json \
  results/plaintext_vs_gl_encrypted_linear_n450.json \
  results/plaintext_vs_gl_encrypted_mlp_n8192.json \
  results/openfhe_numpy_env_probe.json \
  results/openfhe_numpy_ckks_matmul_toy.json \
  results/openfhe_numpy_ckks_matmul_sweep.csv \
  results/openfhe_numpy_failure_summary.md \
  results/gl_vs_openfhe_ctct_matmul_comparison.csv
do
  if [ -f "$f" ]; then
    cp "$f" "$OUT/key_results/"
  fi
done

# Source snippets for report evidence.
if [ -f src/gl_encrypted_weight.py ]; then
  nl -ba src/gl_encrypted_weight.py | sed -n '1,340p' > "$OUT/source_snippets/gl_encrypted_weight_l1_340.txt"
fi

if [ -f src/model_pca32.py ]; then
  nl -ba src/model_pca32.py | sed -n '1,180p' > "$OUT/source_snippets/model_pca32_l1_180.txt"
fi

if [ -f src/polynomial.py ]; then
  nl -ba src/polynomial.py | sed -n '1,160p' > "$OUT/source_snippets/polynomial_l1_160.txt"
fi

if [ -f scripts/09_train_pca32_mlp.py ]; then
  nl -ba scripts/09_train_pca32_mlp.py | sed -n '1,120p' > "$OUT/source_snippets/09_train_pca32_mlp_l1_120.txt"
fi

grep -RIn \
  -e "degree3" \
  -e "poly" \
  -e "evaluate_polynomial" \
  -e "polynomial_radius" \
  -e "expand_eval_split" \
  -e "require_sample_count" \
  src scripts 2>/dev/null \
  > "$OUT/source_keyword_hits.txt" || true

python3 - <<'PY'
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

out_dir = Path("results/report_inputs")
rows = []
level_rows = []

patterns = [
    "results/gl_encrypted_weight_*.json",
    "results/gl_ctct_matrix_multiply_probe.json",
    "results/gl_encrypted_weight_mutation_tests.json",
    "results/pca32_train.json",
    "results/openfhe_numpy_*.json",
]

files = []
for p in patterns:
    files.extend(Path(".").glob(p))
files = sorted(set(files))

def safe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

for path in files:
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue

    if not isinstance(data, dict):
        continue

    total = safe_get(data, "total_s", safe_get(data, "runtime_total_s"))
    keygen = safe_get(data, "key_generation_s")
    server = safe_get(data, "server_only_s")
    n = safe_get(data, "n_samples")

    def div(a, b):
        try:
            if a is None or b in (None, 0):
                return None
            return float(a) / float(b)
        except Exception:
            return None

    no_keygen = None
    try:
        if total is not None and keygen is not None:
            no_keygen = float(total) - float(keygen)
    except Exception:
        pass

    rows.append({
        "file": str(path),
        "ok": safe_get(data, "ok"),
        "semantic_validation_passed": safe_get(data, "semantic_validation_passed"),
        "failure_category": safe_get(data, "failure_category"),
        "failure_detail": safe_get(data, "failure_detail"),
        "task_type": safe_get(data, "task_type"),
        "weight_privacy": safe_get(data, "weight_privacy"),
        "activation": safe_get(data, "activation"),
        "polynomial_degree": safe_get(data, "polynomial_degree"),
        "polynomial_radius": safe_get(data, "polynomial_radius"),
        "logical_dims": json.dumps(safe_get(data, "logical_dims"), ensure_ascii=False),
        "shape": json.dumps(safe_get(data, "shape"), ensure_ascii=False),
        "n_samples": n,
        "ciphertext_count": safe_get(data, "ciphertext_count"),
        "sample_packing_utilization": safe_get(data, "sample_packing_utilization"),
        "key_generation_s": keygen,
        "input_encryption_s": safe_get(data, "input_encryption_s"),
        "weight_encryption_s": safe_get(data, "weight_encryption_s"),
        "server_only_s": server,
        "total_s": total,
        "total_minus_keygen_s": no_keygen,
        "runtime_per_sample_s": safe_get(data, "runtime_per_sample_s"),
        "server_per_sample_s": div(server, n),
        "no_keygen_per_sample_s": div(no_keygen, n),
        "accuracy": safe_get(data, "accuracy"),
        "argmax_agreement": safe_get(data, "argmax_agreement"),
        "relative_l2": safe_get(data, "logits_relative_l2", safe_get(data, "hidden_relative_l2")),
        "linf": safe_get(data, "logits_linf", safe_get(data, "hidden_linf")),
        "mae": safe_get(data, "logits_mae", safe_get(data, "hidden_mae")),
    })

    for item in safe_get(data, "level_log", []) or []:
        if isinstance(item, dict):
            level_rows.append({
                "file": str(path),
                "stage": item.get("stage"),
                "type": item.get("type"),
                "level": item.get("level"),
                "nbytes": item.get("nbytes"),
                "serialized_nbytes": item.get("serialized_nbytes"),
            })

summary_csv = out_dir / "json_result_summary.csv"
if rows:
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

levels_csv = out_dir / "level_summary.csv"
if level_rows:
    with levels_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(level_rows[0].keys()))
        w.writeheader()
        w.writerows(level_rows)

md = out_dir / "report_input_summary.md"
with md.open("w") as f:
    f.write("# Report input summary\n\n")
    f.write("## Key JSON result summary\n\n")
    if not rows:
        f.write("No JSON rows parsed.\n")
    else:
        headers = [
            "file", "task_type", "n_samples", "activation",
            "sample_packing_utilization", "server_only_s",
            "total_s", "total_minus_keygen_s",
            "server_per_sample_s", "no_keygen_per_sample_s",
            "semantic_validation_passed", "relative_l2",
            "accuracy", "argmax_agreement",
        ]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for r in rows:
            vals = []
            for h in headers:
                v = r.get(h)
                if isinstance(v, float):
                    if math.isfinite(v):
                        v = f"{v:.6g}"
                vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")

print(f"wrote {summary_csv}")
print(f"wrote {levels_csv}")
print(f"wrote {md}")
PY

# Try to extract polynomial/model metadata if joblib exists.
if [ -x "./bin/python3" ] && [ -f "data/pca32_mlp.joblib" ]; then
  ./bin/python3 - <<'PY' > results/report_inputs/model_joblib_probe.txt
from __future__ import annotations

import joblib
from pathlib import Path

path = Path("data/pca32_mlp.joblib")
obj = joblib.load(path)

print("joblib:", path)
print("top_type:", type(obj))

def walk(x, prefix="", depth=0):
    if depth > 4:
        return
    if isinstance(x, dict):
        for k, v in x.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            lower = str(k).lower()
            if any(s in lower for s in ["poly", "coeff", "radius", "relu", "pca", "x_test", "base"]):
                try:
                    shape = getattr(v, "shape", None)
                    print(name, "type=", type(v), "shape=", shape, "value=", repr(v)[:500])
                except Exception as e:
                    print(name, "ERR", repr(e))
            walk(v, name, depth + 1)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x[:20]):
            walk(v, f"{prefix}[{i}]", depth + 1)

walk(obj)
PY
else
  echo "skip model_joblib_probe: ./bin/python3 or data/pca32_mlp.joblib not found" \
    > results/report_inputs/model_joblib_probe.txt
fi

tar -czf "results/report_inputs_$(hostname)_$(date +%Y%m%d_%H%M%S).tgz" results/report_inputs

echo "[collect] done"
echo "[collect] summary:"
ls -lh results/report_inputs*
