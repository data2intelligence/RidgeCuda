# RidgeCuda

GPU-accelerated ridge regression with permutation testing, powered by
NVIDIA CUDA. Exports `ridge()` for one-shot inference and
`ridge_batch()` for memory-efficient column-batched inference with
optional HDF5 I/O — an optional accelerator for the
[SecAct](https://github.com/data2intelligence/SecAct) R package.

## When to install

Install RidgeCuda if you have an NVIDIA GPU (Compute Capability 6.0+)
and are running SecAct on large datasets. SecAct works out-of-the-box
without it (pure-R fallback) and falls back to
[RidgeFast](https://github.com/psychemistz/RidgeFast) on CPU. With
RidgeCuda installed and a GPU visible, SecAct auto-dispatches to it
via `backend = "auto"`.

## Requirements

- R (>= 3.6.0)
- NVIDIA CUDA Toolkit (>= 11.0 recommended)
- NVIDIA GPU with compute capability 6.0+ (Pascal or newer) for
  double-precision
- Optional: `rhdf5` for `ridge_batch()` HDF5 input/output

## Installation

```r
# Install dependencies
install.packages("remotes")
install.packages("Matrix")

# Optional, for HDF5 streaming:
# BiocManager::install("rhdf5")

# Install RidgeCuda (needs CUDA_HOME set if it isn't auto-detected)
remotes::install_github("psychemistz/RidgeCuda")
```

## Usage

### One-shot inference (SecAct accelerator contract)

Use `ridge()` as a drop-in replacement for `RidgeFast::ridge()`:

```r
library(RidgeCuda)

# X: n x p signature matrix, column-scaled
# Y: n x m expression matrix, column-scaled
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

### Legacy `ridge_cuda()` API

The lower-level `ridge_cuda()`, sparse `Y` support, and CUDA utilities
(`check_cuda_available`, `get_cuda_devices`, `get_cuda_memory_info`,
`cleanup_cuda`, `scale_dense_matrix_cuda`, `scale_sparse_matrix_csc_cuda`)
remain available for users who need fine-grained control.

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
  to SecAct's pure-R and RidgeFast's CPU-fast backends in their
  canonical modes. Batched output is bit-identical to single-call
  output. Parity is verified by the
  [`tests/test_gpu_batch.sbatch`](tests/test_gpu_batch.sbatch) SLURM
  script.
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
- **Compute-capability too low**: this package requires CC ≥ 6.0 for
  double-precision. Older GPUs (Maxwell, Kepler) are not supported.

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
