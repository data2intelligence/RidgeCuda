# RidgeCuda

GPU-accelerated ridge regression with permutation-based significance
testing, powered by NVIDIA CUDA. Exports `ridge()` for one-shot
inference and `ridge_batch()` for memory-efficient column-batched
inference with optional HDF5 I/O.

Built on cuBLAS, cuSOLVER, cuSPARSE, and cuRAND. Designed for problems
of the form \( Y = X\beta \) where `X` is a feature / signature matrix
shared across many samples (columns of `Y`), and permutation-based
null distributions are needed for significance.

## When to use

`ridge()` and `ridge_batch()` accept any real-valued `X (n × p)` and
`Y (n × m)` where rows are shared observations (e.g., genes) and
columns are features (`p`) and samples (`m`). Use RidgeCuda when you
want:

- Large-scale permutation testing (many samples × many signatures ×
  many permutations) offloaded to a GPU
- A single GPU dependency stack (CUDA ≥ 11.0) — no CPU fallback
  required in the hot path
- Sparse `Y` support via `dgCMatrix` on the lower-level `ridge_cuda()`
  entry point
- Bit-reproducible output matching CPU backends (MT19937 seed 0)

## Companion packages

[RidgeFast](https://github.com/psychemistz/RidgeFast) provides the
same `ridge()` / `ridge_batch()` API on CPU (GSL + OpenMP) and is
drop-in interchangeable with RidgeCuda.

[SecAct](https://github.com/data2intelligence/SecAct) (secreted
protein activity inference) is one application that benefits from
RidgeCuda: at `backend = "auto"` it dispatches to the GPU when a
compatible device is available, gaining a large speed-up on
single-cell and large-cohort inference. RidgeCuda has no dependency
on SecAct and can be used standalone for any ridge regression
workflow.

## Requirements

- R (>= 3.6.0)
- NVIDIA CUDA Toolkit (>= 11.0 recommended)
- NVIDIA GPU with compute capability 6.0+ (Pascal or newer) for
  double precision
- Optional: `rhdf5` for `ridge_batch()` HDF5 input/output

## Installation

```r
install.packages(c("remotes", "Matrix"))
# BiocManager::install("rhdf5")   # optional, for HDF5 streaming

# Install (auto-detects CUDA_HOME; set explicitly if needed)
remotes::install_github("psychemistz/RidgeCuda")
```

## Usage

### One-shot inference

```r
library(RidgeCuda)

# X: n x p feature matrix, column-scaled
# Y: n x m response matrix, column-scaled
res <- ridge(X, Y,
             lambda     = 5e+05,
             nrand      = 1000,
             ncores     = 1L,          # ignored on GPU
             rng_method = "mt19937",   # "mt19937" | "srand"
             device_id  = 0L)
str(res)
# List of 4: beta, se, zscore, pvalue — each p x m matrix
```

### Batched inference (large m)

`ridge_batch()` processes `Y` in column-batches, bounding peak GPU
memory. Output is bit-identical to `ridge()` at `ncores = 1`.

```r
# In-memory Y
res <- ridge_batch(X, Y, lambda = 5e+05, nrand = 1000, batch_size = 5000)

# HDF5 input
res <- ridge_batch(X, Y = "Y_expr.h5",
                   lambda = 5e+05, nrand = 1000, batch_size = 5000)

# Streaming output
meta <- ridge_batch(X, Y, lambda = 5e+05, nrand = 1000,
                    batch_size = 5000, output_h5 = "results.h5")

# Custom reader
res <- ridge_batch(X, Y = NULL, n_samples = m, reader = my_reader,
                   lambda = 5e+05, nrand = 1000, batch_size = 5000)
```

### Lower-level `ridge_cuda()` API

The `ridge_cuda()` entry point and CUDA utility functions
(`check_cuda_available`, `get_cuda_devices`, `get_cuda_memory_info`,
`cleanup_cuda`, `scale_dense_matrix_cuda`,
`scale_sparse_matrix_csc_cuda`) remain available for users who need
fine-grained control over batching, device selection, async mode, or
sparse `Y`.

```r
library(RidgeCuda)
library(Matrix)

check_cuda_available()
get_cuda_devices()

# Sparse Y support (dgCMatrix)
Y_sparse <- Matrix(Y * (runif(length(Y)) > 0.95), sparse = TRUE)
res <- ridge_cuda(X, Y_sparse, lambda = 1.0, n_rand = 1000)
summary(res)

cleanup_cuda()
```

## Reproducibility

- `rng_method = "mt19937"` (default) with `ncores = 1` → bit-identical
  to `RidgeFast::ridge()` in its canonical mode. Batched output is
  bit-identical to single-call output. Parity is verified by
  [`tests/test_gpu_batch.sbatch`](tests/test_gpu_batch.sbatch).
- `rng_method = "srand"` → faster setup but not reproducible.

## API

```
ridge(X, Y, lambda, nrand, ncores, rng_method, device_id)
ridge_batch(X, Y, lambda, nrand, ncores, rng_method, device_id,
            batch_size, reader, n_samples, output_h5, verbose)
```

Identical to `RidgeFast` except for the `device_id` argument.

## Troubleshooting

- **CUDA not found at install**: set `CUDA_HOME` to your toolkit
  install path (e.g., `/usr/local/CUDA/12.1.0`) before
  `remotes::install_github`.
- **GPU out-of-memory**: reduce `batch_size` in `ridge_batch()`, or
  lower `nrand`; check `get_cuda_memory_info()` for headroom.
- **Compute capability too low**: this package requires CC ≥ 6.0 for
  double precision. Older GPUs (Maxwell, Kepler) are not supported.

## Citation

```
@misc{RidgeCuda,
  author = {Seongyong Park},
  title = {RidgeCuda: CUDA-Accelerated Ridge Regression for R},
  url = {https://github.com/psychemistz/RidgeCuda}
}
```

## License

GPL (>= 2)
