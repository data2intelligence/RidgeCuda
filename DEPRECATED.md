# RidgeCuda is deprecated — please use FlashReg

**The RidgeCuda R package has been superseded by
[FlashReg](https://github.com/data2intelligence/FlashReg).**

FlashReg is one R package that ships the same CUDA GPU kernel as
RidgeCuda *and* the C+OpenMP CPU kernel formerly provided by
[RidgeFast](https://github.com/data2intelligence/RidgeFast), with a
single unified `ridge()` entry point and `backend="auto"` resolution.

The kernel sources, device-management helpers, and the MT19937
permutation seam are identical to those that lived here, so GPU
results are bit-equivalent to RidgeCuda at `rng_method="mt19937"`.

## Migration in one line

```r
# Before
RidgeCuda::ridge(X, Y, lambda, nrand, rng_method = "mt19937")

# After
FlashReg::ridge(X, Y, lambda, nrand, backend = "cuda_native",
                rng_method = "mt19937")

# Or let FlashReg pick — it auto-resolves to cuda_native when a
# usable GPU is detected at runtime.
FlashReg::ridge(X, Y, lambda, nrand, backend = "auto",
                rng_method = "mt19937")
```

## CUDA build is optional

FlashReg's `configure` detects `nvcc` at install time. If absent (or
if you set `FLASHREG_NO_CUDA=1`), the package builds CPU-only with the
OMP backend. At runtime, `FlashReg::cuda_available()` returns FALSE on
machines without a usable GPU even when the toolkit was present at
build time, and `backend="auto"` falls back to `omp` cleanly.

## Where the code went

| Was in RidgeCuda | Is now in FlashReg |
|---|---|
| `src/ridge_cuda.cu` | `inst/cuda/ridge_cuda.cu` |
| `src/ridge_cuda.h` | `inst/cuda/ridge_cuda.h` |
| `src/ridge_r_interface.cpp` | `src/flashreg_shim_cuda.c` (rewritten as plain C) |
| `RidgeCuda::ridge()` | `FlashReg::ridge(backend="cuda_native")` |
| `RidgeCuda::check_cuda_available()` | `FlashReg::cuda_available()` (with runtime device probe) |
| `RidgeCuda::ridge_batch()` | `FlashReg::ridge()` called per batch (caller-side loop) |

This repository remains read-only for archival reference. No further
releases or bug fixes will be made here — file issues against
FlashReg instead.
