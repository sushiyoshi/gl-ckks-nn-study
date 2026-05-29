#!/usr/bin/env bash
set -euo pipefail

python3 scripts/26_run_large_batch_packed_pca32.py --modes gl lightplainmatrix --samples 450 1024 2048 4096 8192
python3 scripts/27_summarize_steady_state.py
git status --short --untracked-files=all
