# Packed GL / CKKS Reproduction Runbook

## Purpose

This repository benchmarks packed GL and CKKS inference on M2 Mac CPU using `desilofhe`.
The focus is:

- GL packed PCA32 inference
- CKKS packed PCA32 inference
- offline / online timing separation
- object size probing
- steady-state throughput
- large-batch scaling

This runbook lists the recommended execution order, expected outputs, and the split between light and heavy runs.

## Prerequisites

- macOS ARM64 on an M2-class Mac
- Python 3.10+
- CPU-only `desilofhe`
- working `data/` inputs already present in the repository
- `results/` treated as generated output and ignored by git

## Script Inventory

Current experiment scripts:

- `scripts/00_check_env.py`
- `scripts/01_train_plain_mlp.py`
- `scripts/02_fit_relu_polynomial.py`
- `scripts/03_run_plain_baselines.py`
- `scripts/04_run_ckks_poly_inference.py`
- `scripts/05_check_gl_capabilities.py`
- `scripts/06_run_gl_poly_inference.py`
- `scripts/07_summarize_results.py`
- `scripts/08_probe_gl_layout.py`
- `scripts/09_train_pca32_mlp.py`
- `scripts/10_run_gl_native_toy.py`
- `scripts/11_run_gl_padded_pca32_classifier.py`
- `scripts/12_run_gl_blocked_digits64_mlp.py`
- `scripts/13_run_ckks_pca32_classifier.py`
- `scripts/14_summarize_layout_overhead.py`
- `scripts/15_run_gl_padded_pca32_packed.py`
- `scripts/16_run_gl_blocked_digits64_packed.py`
- `scripts/17_compare_single_vs_packed_gl.py`
- `scripts/18_run_ckks_pca32_packed.py`
- `scripts/19_compare_gl_ckks_packed.py`
- `scripts/20_probe_ckks_matrix_api.py`
- `scripts/21_run_ckks_pca32_packed_sweep.py`
- `scripts/22_run_gl_pca32_packed_sweep.py`
- `scripts/23_compare_packed_sweep.py`
- `scripts/24_measure_object_sizes.py`
- `scripts/25_run_steady_state_packed_pca32.py`
- `scripts/26_run_large_batch_packed_pca32.py`
- `scripts/27_summarize_steady_state.py`

## Execution Order

### 1. Environment / API Probe

```bash
python3 scripts/00_check_env.py
python3 scripts/05_check_gl_capabilities.py
python3 scripts/08_probe_gl_layout.py
python3 scripts/20_probe_ckks_matrix_api.py
```

Outputs:

- `results/env.json`
- `results/gl_capabilities.json`
- `results/gl_layout_probe.json`
- `results/ckks_matrix_api_probe.json`

### 2. Training / Model Preparation

```bash
python3 scripts/09_train_pca32_mlp.py
python3 scripts/02_fit_relu_polynomial.py
python3 scripts/01_train_plain_mlp.py
```

Outputs:

- `data/pca32_mlp.joblib`
- `data/plain_mlp.joblib`
- `data/relu_poly_degree3.npz`
- `results/pca32_train.json`
- `results/train_plain_mlp.json`
- `results/relu_polynomial.json`

### 3. GL Single and Packed Validation

```bash
python3 scripts/10_run_gl_native_toy.py --degree 3
python3 scripts/11_run_gl_padded_pca32_classifier.py --degree 3 --n-samples 1
python3 scripts/12_run_gl_blocked_digits64_mlp.py --degree 3 --n-samples 1
python3 scripts/15_run_gl_padded_pca32_packed.py --degree 3 --n-samples 32
python3 scripts/15_run_gl_padded_pca32_packed.py --degree 3 --n-samples 450
python3 scripts/16_run_gl_blocked_digits64_packed.py --degree 3 --n-samples 32
python3 scripts/16_run_gl_blocked_digits64_packed.py --degree 3 --n-samples 450
python3 scripts/17_compare_single_vs_packed_gl.py
```

Outputs:

- `results/gl_native_toy.json`
- `results/gl_native_toy.csv`
- `results/gl_padded_pca32_results.json`
- `results/gl_padded_pca32_results.csv`
- `results/gl_blocked_digits64_results.json`
- `results/gl_blocked_digits64_results.csv`
- `results/gl_padded_pca32_packed_results.json`
- `results/gl_padded_pca32_packed_results.csv`
- `results/gl_blocked_digits64_packed_results.json`
- `results/gl_blocked_digits64_packed_results.csv`
- `results/gl_packing_summary.csv`

### 4. CKKS Packed Validation

```bash
python3 scripts/13_run_ckks_pca32_classifier.py --degree 3 --n-samples 1
python3 scripts/18_run_ckks_pca32_packed.py --degree 3 --n-samples 32
python3 scripts/18_run_ckks_pca32_packed.py --degree 3 --n-samples 450
python3 scripts/19_compare_gl_ckks_packed.py
```

Outputs:

- `results/ckks_pca32_results.json`
- `results/ckks_pca32_results.csv`
- `results/ckks_pca32_packed_results.json`
- `results/ckks_pca32_packed_results.csv`
- `results/gl_ckks_packed_comparison.csv`

### 5. Packed Sweep

```bash
python3 scripts/21_run_ckks_pca32_packed_sweep.py --mode lightplainmatrix
python3 scripts/22_run_gl_pca32_packed_sweep.py
python3 scripts/23_compare_packed_sweep.py
```

Outputs:

- `results/ckks_pca32_packed_sweep_lightplainmatrix.csv`
- `results/ckks_pca32_packed_sweep.csv`
- `results/gl_pca32_packed_sweep.csv`
- `results/packed_sweep_comparison.csv`

### 6. Steady-State Measurement

```bash
python3 scripts/25_run_steady_state_packed_pca32.py
python3 scripts/27_summarize_steady_state.py
```

Outputs:

- `results/steady_state_packed_pca32.json`
- `results/steady_state_packed_pca32.csv`
- `results/final_packed_pca32_summary.csv`
- `results/final_packed_pca32_summary.md`

### 7. Object Size Measurement

```bash
python3 scripts/24_measure_object_sizes.py
```

Outputs:

- `results/object_sizes.json`
- `results/object_sizes.csv`

### 8. RSS Memory Probe

Not currently implemented as a dedicated script in this repository.
If added later, place it after the object size probe and before the final summary.

### 9. Final Summary

```bash
python3 scripts/14_summarize_layout_overhead.py
python3 scripts/27_summarize_steady_state.py
```

Outputs:

- `results/layout_overhead_summary.csv`
- `results/final_packed_pca32_summary.csv`
- `results/final_packed_pca32_summary.md`

## Light vs Heavy Runs

Lightweight runs:

- environment / API probes
- training / model preparation
- single-sample GL validation
- 32-sample packed validation
- object size probe
- steady-state 450-sample measurement

Heavy runs:

- large-batch sweep with `1024`, `2048`, `4096`, `8192` samples
- any CKKS dense path at high sample counts

The heavy sweep is intentionally separated into its own shell wrapper.

## Semantic Validation

Treat a run as semantically successful only if the script reports:

- `semantic_validation_passed = true`
- and the reported error metric is acceptable, typically `relative_l2 < 1e-5`

For packed runs, compare against the matching plaintext polynomial baseline with the same sample order.

## Git-Managed vs Generated Files

Generated files live under `data/` and `results/` and are ignored by git in this repository setup.
Do not add virtualenv directories, `site-packages`, or large generated artifacts to version control.

## Key Result Files to Inspect

- `results/gl_layout_probe.json`
- `results/ckks_matrix_api_probe.json`
- `results/pca32_train.json`
- `results/gl_native_toy.json`
- `results/gl_padded_pca32_packed_results.json`
- `results/gl_blocked_digits64_packed_results.json`
- `results/ckks_pca32_packed_results.json`
- `results/packed_sweep_comparison.csv`
- `results/object_sizes.json`
- `results/steady_state_packed_pca32.csv`
- `results/large_batch_packed_pca32.csv`
- `results/final_packed_pca32_summary.csv`
- `results/final_packed_pca32_summary.md`
