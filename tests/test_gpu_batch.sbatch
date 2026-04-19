#!/bin/bash
#SBATCH --job-name=ridgecuda-batch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=test_gpu_batch.%j.log

# ridge_batch parity test: verify batched output matches single-call
# output on GPU, and matches RidgeFast::ridge_batch, and matches HDF5
# round-trip (write + read).

set -euo pipefail

module load CUDA/12.1 gcc/11.3.0 GSL/2.7_gcc-11.3.0 R/4.3.2

cd /vf/users/parks34/projects/1ridgesig/RidgeCuda
R CMD INSTALL .

cd /vf/users/parks34/projects/1ridgesig

R --no-save <<'EOF'
library(RidgeFast)
library(RidgeCuda)

set.seed(23)
n <- 300; p <- 15; m <- 140
X <- matrix(rnorm(n * p), n, p); colnames(X) <- paste0("sig", 1:p)
Y <- matrix(rnorm(n * m), n, m); colnames(Y) <- paste0("s", 1:m)
X <- scale(X); Y <- scale(Y)

lambda <- 50; nrand <- 150L; bs <- 33L

cat("\n--- GPU single-call ---\n")
r_single <- RidgeCuda::ridge(X, Y, lambda, nrand, ncores = 1L,
                             rng_method = "mt19937")

cat("\n--- GPU batched (in-memory) ---\n")
r_bmem <- RidgeCuda::ridge_batch(X, Y, lambda, nrand, ncores = 1L,
                                 rng_method = "mt19937",
                                 batch_size = bs)

cat("\n=== PARITY: GPU single vs GPU batched ===\n")
for (nm in c("beta", "se", "zscore", "pvalue")) {
  d <- max(abs(r_single[[nm]] - r_bmem[[nm]]))
  status <- if (d < 1e-9) "PASS" else if (d < 1e-6) "NEAR" else "FAIL"
  cat(sprintf("  %-7s max|diff| = %.3e   [%s]\n", nm, d, status))
}

cat("\n--- GPU batched (HDF5 in/out) ---\n")
h5_in  <- tempfile(fileext = ".h5")
h5_out <- tempfile(fileext = ".h5")
rhdf5::h5createFile(h5_in)
rhdf5::h5write(Y, h5_in, "Y")
rhdf5::h5write(colnames(Y), h5_in, "sample_names")
meta <- RidgeCuda::ridge_batch(X, Y = h5_in, lambda, nrand, ncores = 1L,
                               rng_method = "mt19937", batch_size = bs,
                               output_h5 = h5_out)
cat(sprintf("  wrote: %s (p=%d, m=%d, %d batches)\n",
            meta$path, meta$p, meta$m, meta$num_batches))

cat("\n=== PARITY: GPU in-mem batched vs GPU HDF5 round-trip ===\n")
for (nm in c("beta", "se", "zscore", "pvalue")) {
  v_h5 <- rhdf5::h5read(h5_out, nm)
  v_mem <- r_bmem[[nm]]; dimnames(v_mem) <- NULL
  d <- max(abs(v_h5 - v_mem))
  status <- if (d < 1e-12) "PASS" else "FAIL"
  cat(sprintf("  %-7s max|diff| = %.3e   [%s]\n", nm, d, status))
}

cat("\n--- RidgeFast batched for cross-backend parity ---\n")
r_fbatch <- RidgeFast::ridge_batch(X, Y, lambda, nrand, ncores = 1L,
                                   rng_method = "mt19937", batch_size = bs)

cat("\n=== PARITY: RidgeFast batched vs RidgeCuda batched ===\n")
for (nm in c("beta", "se", "zscore", "pvalue")) {
  d <- max(abs(r_fbatch[[nm]] - r_bmem[[nm]]))
  status <- if (d < 1e-9) "PASS" else if (d < 1e-6) "NEAR" else "FAIL"
  cat(sprintf("  %-7s max|diff| = %.3e   [%s]\n", nm, d, status))
}

unlink(h5_in); unlink(h5_out)
cat("\nDone.\n")
EOF
