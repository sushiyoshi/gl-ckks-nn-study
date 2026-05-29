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
- encrypted-weight GL ciphertext-ciphertext matrix multiplication validation
- external OpenFHE-NumPy CKKS ciphertext-ciphertext matrix multiplication baseline

This runbook lists the recommended execution order, expected outputs, and the split between light and heavy runs.

## Prerequisites

- macOS ARM64 on an M2-class Mac
- Python 3.10+
- CPU-only `desilofhe`
- OpenFHE-NumPy is treated as an external baseline and uses its own isolated environment
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
- `scripts/30_probe_gl_ctct_matrix_multiply.py`
- `scripts/31_run_gl_encrypted_weight_pca32_linear.py`
- `scripts/32_run_gl_encrypted_weight_pca32_two_linear.py`
- `scripts/33_run_gl_encrypted_weight_pca32_mlp.py`
- `scripts/34_compare_gl_plain_vs_encrypted_weight.py`
- `scripts/35_run_gl_encrypted_weight_mutation_tests.py`
- `scripts/36_probe_openfhe_numpy_env.py`
- `scripts/37_run_openfhe_numpy_ckks_matmul_toy.py`
- `scripts/38_run_openfhe_numpy_ckks_matmul_sweep.py`
- `scripts/39_compare_gl_openfhe_ctct_matmul.py`

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

### 7. Encrypted-Weight GL Validation

These runs use a different threat model from the existing plaintext-weight GL packed PCA32 inference: both weights and inputs are encrypted to test model privacy feasibility.  Validate ciphertext-ciphertext matrix multiplication before running full inference.

```bash
python3 scripts/30_probe_gl_ctct_matrix_multiply.py
python3 scripts/31_run_gl_encrypted_weight_pca32_linear.py --n-samples 1
python3 scripts/31_run_gl_encrypted_weight_pca32_linear.py --n-samples 32
python3 scripts/31_run_gl_encrypted_weight_pca32_linear.py --n-samples 450
python3 scripts/32_run_gl_encrypted_weight_pca32_two_linear.py --n-samples 1
python3 scripts/32_run_gl_encrypted_weight_pca32_two_linear.py --n-samples 32
python3 scripts/32_run_gl_encrypted_weight_pca32_two_linear.py --n-samples 450
python3 scripts/33_run_gl_encrypted_weight_pca32_mlp.py --degree 3 --n-samples 1
python3 scripts/33_run_gl_encrypted_weight_pca32_mlp.py --degree 3 --n-samples 32
python3 scripts/34_compare_gl_plain_vs_encrypted_weight.py
```

If the run becomes too heavy, keep the 450-sample run to the linear-only script and let two-linear or polynomial MLP failures be recorded in JSON.  A degree-3 polynomial activation after ciphertext-weight multiplication may fail from level exhaustion or scale mismatch; that is a meaningful result.

Outputs:

- `results/gl_ctct_matrix_multiply_probe.json`
- `results/gl_encrypted_weight_pca32_linear_n*.json`
- `results/gl_encrypted_weight_pca32_two_linear_n*.json`
- `results/gl_encrypted_weight_pca32_mlp_n*.json`
- `results/gl_plain_vs_encrypted_weight_comparison.csv`
- `results/gl_plain_vs_encrypted_weight_comparison.md`

OpenFHE/CKKS rotation-based ciphertext-ciphertext matrix multiplication is not included in this GL validation track.

### 8. OpenFHE-NumPy External CKKS Baseline

This baseline is intentionally isolated from the existing `desilofhe` environment. Do not modify `./bin/python3`, `pyvenv/`, `.venv/`, `lib/`, `include/`, or any existing site-packages tree when trying it.

Native macOS attempt:

```bash
python3 -m venv .openfhe_numpy_venv
source .openfhe_numpy_venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install openfhe_numpy numpy pandas
python scripts/36_probe_openfhe_numpy_env.py
python scripts/37_run_openfhe_numpy_ckks_matmul_toy.py --d 2
python scripts/38_run_openfhe_numpy_ckks_matmul_sweep.py
python scripts/39_compare_gl_openfhe_ctct_matmul.py
```

Failure classes to record in `results/openfhe_numpy_env_probe.json` and the toy/sweep JSON files:

- `wheel unsupported`
- `build dependency missing`
- `openfhe missing`
- `import error`
- `api mismatch`

If native macOS install or import fails, move to the Ubuntu fallback instead of changing the existing GL environment.

Ubuntu / Docker fallback:

```bash
bash scripts/run_openfhe_numpy_docker.sh
```

Inside the container:

```bash
python scripts/36_probe_openfhe_numpy_env.py
python scripts/37_run_openfhe_numpy_ckks_matmul_toy.py --d 2
python scripts/38_run_openfhe_numpy_ckks_matmul_sweep.py
python scripts/39_compare_gl_openfhe_ctct_matmul.py
```

Fallback container policy:

- base image: `ubuntu:24.04`
- install: `python3`, `python3-venv`, `python3-pip`, `git`, `build-essential`, `cmake`
- venv path: `/opt/openfhe_numpy_venv`
- mount the repo at `/work`
- write results back into `/work/results`

OpenFHE-NumPy current limitation note:

- the current implementation is centered on single-ciphertext vector/matrix layouts
- block ciphertext support is not the primary path yet and should be treated as future work
- therefore the comparison to GL is an external CKKS ct-ct baseline, not a fully layout-matched benchmark

Next candidate if OpenFHE-NumPy is not usable:

- OpenFHE `polyakov-matrix-mult` branch

### 9. Object Size Measurement

```bash
python3 scripts/24_measure_object_sizes.py
```

Outputs:

- `results/object_sizes.json`
- `results/object_sizes.csv`

### 10. RSS Memory Probe

Not currently implemented as a dedicated script in this repository.
If added later, place it after the object size probe and before the final summary.

### 11. Final Summary

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
- encrypted-weight GL two-linear or polynomial MLP runs after the 32-sample check
- any CKKS dense path at high sample counts

The heavy sweep is intentionally separated into its own shell wrapper.

## Semantic Validation

Treat a run as semantically successful only if the script reports:

- `semantic_validation_passed = true`
- and the reported error metric is acceptable, typically `relative_l2 < 1e-5`

For packed runs, compare against the matching plaintext polynomial baseline with the same sample order.

For encrypted-weight GL runs, validate in this order: `ct-ct matrix_multiply` toy probe, encrypted `W1 x X` linear layer, encrypted `W1/W2` two-linear inference, then degree-3 polynomial MLP.  Do not interpret an MLP failure as invalidating linear-only ciphertext-ciphertext matrix multiplication unless the earlier semantic checks also fail.

## Git-Managed vs Generated Files

Generated files live under `data/` and `results/` and are ignored by git in this repository setup.
Do not add virtualenv directories, `site-packages`, or large generated artifacts to version control.

## Key Result Files to Inspect

- `results/gl_layout_probe.json`
- `results/ckks_matrix_api_probe.json`
- `results/openfhe_numpy_env_probe.json`
- `results/openfhe_numpy_ckks_matmul_toy.json`
- `results/openfhe_numpy_ckks_matmul_sweep.csv`
- `results/gl_vs_openfhe_ctct_matmul_comparison.csv`
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
