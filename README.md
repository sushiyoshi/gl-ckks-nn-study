# desilofhe CKKS vs GL polynomial-ReLU MLP smoke study

This directory contains a small CPU-only experiment for Apple Silicon/macOS.  It compares only:

- A. CKKS linear layers with plaintext weights plus polynomial ReLU.
- B. GL linear layers with plaintext weights plus polynomial ReLU, only if the public `desilofhe` API supports that threat model.

It intentionally does not cover functional bootstrapping, RBOOT, ordinary bootstrapping, GPU/CUDA, or encrypted training.

The GL experiments should be read as layout and overhead measurements.  GL is used as fixed-shape batched matrix arithmetic, not as a scheme that automatically accepts arbitrary vector MLP layers without blocking or padding.

## Dataset and Model

- Dataset: `sklearn.datasets.load_digits` (8x8 handwritten digits, not MNIST).
- Input dimension: 64.
- Classes: 10.
- First model: `64 -> 16 -> 10`.
- PCA32 model: `32 -> 32 -> 10`, where PCA is fitted on the training split.  This is GL-friendly and is not evidence that a general MLP maps naturally to GL.
- GL-native toy model: synthetic `32 -> 32 -> 32`.
- Weights stay plaintext. Inputs are encrypted in FHE runs.
- ReLU is replaced by a fitted power-basis polynomial, degree 3 by default.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

This environment was checked with Python 3.13.4 and `desilofhe` 1.13.0.  Python 3.10+ is the target.

## Run

Run in small steps.  Start with the environment and capability checks:

```bash
python3 scripts/00_check_env.py
python3 scripts/05_check_gl_capabilities.py
python3 scripts/08_probe_gl_layout.py
python3 scripts/09_train_pca32_mlp.py
```

Train and evaluate plaintext baselines:

```bash
python3 scripts/01_train_plain_mlp.py --hidden 16
python3 scripts/02_fit_relu_polynomial.py --degree 3
python3 scripts/03_run_plain_baselines.py --degree 3
```

Run FHE smoke tests.  Start with one or a few samples on M2 before increasing to 10 or 100:

```bash
python3 scripts/04_run_ckks_poly_inference.py --degree 3 --n-samples 1
python3 scripts/06_run_gl_poly_inference.py --degree 3 --n-samples 1
python3 scripts/07_summarize_results.py
```

Layout and overhead experiments:

```bash
python3 scripts/10_run_gl_native_toy.py --degree 3
python3 scripts/11_run_gl_padded_pca32_classifier.py --degree 3 --n-samples 1
python3 scripts/12_run_gl_blocked_digits64_mlp.py --degree 3 --n-samples 1
python3 scripts/13_run_ckks_pca32_classifier.py --degree 3 --n-samples 1
python3 scripts/14_summarize_layout_overhead.py
```

Packed GL throughput experiments:

```bash
python3 scripts/15_run_gl_padded_pca32_packed.py --degree 3 --n-samples 32
python3 scripts/15_run_gl_padded_pca32_packed.py --degree 3 --n-samples 450
python3 scripts/16_run_gl_blocked_digits64_packed.py --degree 3 --n-samples 32
python3 scripts/16_run_gl_blocked_digits64_packed.py --degree 3 --n-samples 450
python3 scripts/17_compare_single_vs_packed_gl.py
```

## Fairness Condition

The intended comparison is only fair if both CKKS and GL can run the same threat model:

- input ciphertext,
- plaintext model weights,
- plaintext bias,
- polynomial activation over ciphertext.

`GLEngine.matrix_multiply` is checked explicitly for `numpy/plaintext x GLCiphertext`.  If this path fails, `scripts/06_run_gl_poly_inference.py` writes `results/unsupported_gl_plain_weight.md` and records only a GL ciphertext-ciphertext matrix multiplication microbenchmark.

For GL layout semantics, run `scripts/08_probe_gl_layout.py` first.  The probe checks `engine.shape`, `encode/decode`, batch 0 and batch 1 matrix multiplication, and a 32->32 toy linear layer.  On `desilofhe==1.13.0`, raw `numpy.ndarray` left operands may be accepted by `matrix_multiply` without matching normal matrix multiplication semantics.  Encoded `GLPlaintext` weights are the validated path, so GL plaintext weights are always passed as `engine.encode(weight_tensor)`.

The blocked `64 -> 16 -> 10` experiment is the main overhead measurement for placing an ordinary load_digits MLP on GL's fixed `32 x 32` matrix slots.  The PCA32 classifier is intentionally GL-friendly and should not be interpreted as a general MLP result.

Single-sample GL inference places one sample in one column of one `32 x 32` matrix, so it mostly behaves like matrix-vector inference and does not use GL's matrix-matrix primitive well.  Packed GL inference uses columns as the sample axis: one `32 x 32` matrix holds up to 32 samples, and the `GLEngine.shape=(256, 32, 32)` batch axis gives a theoretical capacity of `256 * 32 = 8192` samples per ciphertext.  The relevant GL comparison is therefore amortized throughput, sample packing utilization, matrix entry utilization, and blocking overhead, not just single-sample latency.

## Result Files

- `results/env.json`
- `results/gl_capabilities.json`
- `results/plain_baseline.csv`
- `results/ckks_results.csv` and `results/ckks_results.json`
- `results/gl_results.csv` and `results/gl_results.json`
- `results/summary.csv`
- `results/gl_layout_probe.json`
- `results/pca32_train.json`
- `results/gl_native_toy.json` and `results/gl_native_toy.csv`
- `results/gl_padded_pca32_results.json` and `results/gl_padded_pca32_results.csv`
- `results/gl_blocked_digits64_results.json` and `results/gl_blocked_digits64_results.csv`
- `results/ckks_pca32_results.json` and `results/ckks_pca32_results.csv`
- `results/layout_overhead_summary.csv`
- `results/gl_padded_pca32_packed_results_n32.json` and `results/gl_padded_pca32_packed_results_n450.json`
- `results/gl_blocked_digits64_packed_results_n32.json` and `results/gl_blocked_digits64_packed_results_n450.json`
- `results/gl_packing_summary.csv`
- `results/unsupported_gl_plain_weight.md` when GL plaintext-weight inference cannot be made comparable

## Notes

CKKS `Engine.multiply_matrix` requires a matrix shaped to the CKKS slot count.  With the default engine this is `8192 x 8192`, so the implementation pads the 64-dimensional input and uses a final constant slot for bias.  This is correct for a small smoke test but memory-heavy.

GL `GLEngine.shape` is recorded and used for zero padding.  The script treats public API failure or shape/semantics mismatch as unsupported rather than forcing an unfair comparison.  The key overheads to watch are output padding for `32 -> 10`, hidden padding for `64 -> 16 -> 10`, and the extra block matrix multiplication needed to split a 64-dimensional input into two 32-dimensional GL columns.
