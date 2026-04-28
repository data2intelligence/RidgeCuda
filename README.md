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

[RidgeFast](https://github.com/data2intelligence/RidgeFast) provides
the same `ridge()` / `ridge_batch()` API on CPU (GSL + OpenMP) and is
drop-in interchangeable with RidgeCuda. Use it on macOS or any host
without an NVIDIA GPU.

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

## Platform support

| OS | Status | Notes |
|---|---|---|
| Linux + NVIDIA GPU | Supported | Tested on RHEL 8 with CUDA 11.x / 12.x |
| Windows + NVIDIA GPU | Supported | Requires CUDA Toolkit + Rtools + MSVC Build Tools (see below) |
| macOS | Not supported | NVIDIA discontinued macOS CUDA after CUDA 10.2 (2019). Use [RidgeFast](https://github.com/data2intelligence/RidgeFast) instead. |

## Installation

### Linux

```r
install.packages(c("remotes", "Matrix"))
# BiocManager::install("rhdf5")   # optional, for HDF5 streaming

# Install (auto-detects CUDA_HOME; set explicitly if needed)
remotes::install_github("data2intelligence/RidgeCuda")
```

If the configure script can't locate the toolkit, set `CUDA_HOME`
(e.g. `/usr/local/cuda-12.1`) before running `install_github`.

### Windows

Install the prerequisites once:

1. R (>= 4.0) and **Rtools** matching your R version
   (https://cran.r-project.org/bin/windows/Rtools/).
2. **NVIDIA CUDA Toolkit** (>= 11.0)
   (https://developer.nvidia.com/cuda-toolkit). The installer sets
   `CUDA_PATH` automatically.
3. **Microsoft C++ Build Tools** (MSVC `cl.exe`). NVIDIA `nvcc` on
   Windows requires MSVC as its host compiler — Rtools' MinGW alone is
   not sufficient for the CUDA translation unit. Install the
   "Desktop development with C++" workload from
   https://visualstudio.microsoft.com/downloads/, and ensure
   `cl.exe` is reachable from a Developer Command Prompt.

Then, from an R session launched out of an environment where both
Rtools and the MSVC toolchain are visible:

```r
install.packages(c("remotes", "Matrix"))
remotes::install_github("data2intelligence/RidgeCuda")
```

`configure.win` honours `CUDA_HOME` first, then `CUDA_PATH`, then
searches the default install root. Override the GPU architectures with
`CUDA_ARCH` if you want a slimmer binary (e.g.
`Sys.setenv(CUDA_ARCH = "-gencode arch=compute_86,code=sm_86")`).

### macOS

Not supported. Install [RidgeFast](https://github.com/data2intelligence/RidgeFast)
for the same `ridge()` / `ridge_batch()` API on CPU.

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

Determinism has two axes: **within a machine** (same binary + same
input → same output) and **across machines / backends** (same input
produces the same output on CPU or on a different GPU / OS).

- `rng_method = "mt19937"` (default). The host builds a GSL MT19937
  (seed 0) permutation table and uploads it to the GPU; the kernel
  consumes these permutations deterministically. Bit-identical to
  `RidgeFast::ridge()` and SecAct's pure-R backend in their canonical
  modes. Parity is verified by
  [`tests/test_gpu_batch.sbatch`](tests/test_gpu_batch.sbatch).
- `rng_method = "srand"`. The kernel runs an in-process Fisher-Yates
  using C standard library `rand()`. Deterministic on a given machine,
  but the permutation sequence depends on the host C runtime, so
  output will *not* match across different OS / compiler combinations,
  and will *not* match the MT19937-based backends.
- GPU floating-point accumulation order may differ by one ULP from the
  CPU path depending on kernel block size; β / se / zscore agree to a
  few ULPs, p-values agree exactly.
- `ridge_batch()` is bit-identical to `ridge()` at the same
  `rng_method` (independent of `batch_size`).

## API

```
ridge(X, Y, lambda, nrand, ncores, rng_method, device_id)
ridge_batch(X, Y, lambda, nrand, ncores, rng_method, device_id,
            batch_size, reader, n_samples, output_h5, verbose)
```

Identical to `RidgeFast` except for the `device_id` argument.

## Using on NIH Biowulf

Install and run on a GPU compute node (CUDA toolchain is not available
on login nodes). Replace `v100x` with `a100`, `p100`, or `v100`
depending on availability (`freen | grep gpu`).

```sh
sinteractive --gres=gpu:v100x:1 --cpus-per-task=8 --mem=16g
module load CUDA/12.1 gcc/11.3.0 R/4.3.2
R -e 'remotes::install_github("data2intelligence/RidgeCuda")'
```

## Troubleshooting

- **CUDA not found at install**: set `CUDA_HOME` to your toolkit
  install path (e.g., `/usr/local/CUDA/12.1.0`) before
  `remotes::install_github`.
- **GPU out-of-memory**: reduce `batch_size` in `ridge_batch()`, or
  lower `nrand`; check `get_cuda_memory_info()` for headroom.
- **Compute capability too low**: this package requires CC ≥ 6.0 for
  double precision. Older GPUs (Maxwell, Kepler) are not supported.

## License

GPL (>= 2)
