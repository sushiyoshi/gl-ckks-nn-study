#!/usr/bin/env bash
set -euo pipefail

python3 -m compileall src scripts
python3 scripts/08_probe_gl_layout.py
python3 scripts/09_train_pca32_mlp.py
python3 scripts/10_run_gl_native_toy.py --degree 3
python3 scripts/11_run_gl_padded_pca32_classifier.py --degree 3 --n-samples 1
python3 scripts/12_run_gl_blocked_digits64_mlp.py --degree 3 --n-samples 1
python3 scripts/15_run_gl_padded_pca32_packed.py --degree 3 --n-samples 32
python3 scripts/15_run_gl_padded_pca32_packed.py --degree 3 --n-samples 450
python3 scripts/16_run_gl_blocked_digits64_packed.py --degree 3 --n-samples 32
python3 scripts/16_run_gl_blocked_digits64_packed.py --degree 3 --n-samples 450
python3 scripts/17_compare_single_vs_packed_gl.py
python3 scripts/18_run_ckks_pca32_packed.py --degree 3 --n-samples 32
python3 scripts/18_run_ckks_pca32_packed.py --degree 3 --n-samples 450
python3 scripts/19_compare_gl_ckks_packed.py
python3 scripts/20_probe_ckks_matrix_api.py
python3 scripts/21_run_ckks_pca32_packed_sweep.py --mode lightplainmatrix
python3 scripts/22_run_gl_pca32_packed_sweep.py
python3 scripts/23_compare_packed_sweep.py
python3 scripts/24_measure_object_sizes.py
python3 scripts/25_run_steady_state_packed_pca32.py
python3 scripts/27_summarize_steady_state.py
git status --short --untracked-files=all
