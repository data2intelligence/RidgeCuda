#!/bin/bash
#SBATCH --job-name=ridgefast-parity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=test_gpu_parity.%j.log

# GPU parity test: verify RidgeCuda with rng_method="mt19937"
# produces output bit-identical (up to FP epsilon) to RidgeFast
# and SecAct pure-R.

set -euo pipefail

module load CUDA/12.1 gcc/11.3.0 GSL/2.7_gcc-11.3.0 R/4.3.2

cd /vf/users/parks34/projects/1ridgesig/RidgeCuda

# Build from source on GPU node (needs nvcc)
R CMD INSTALL .

cd /vf/users/parks34/projects/1ridgesig

R --no-save <<'EOF'
library(RidgeFast)
library(RidgeCuda)

set.seed(17)
n <- 200; p <- 20; m <- 8
X <- matrix(rnorm(n * p), n, p); colnames(X) <- paste0("sig", 1:p)
Y <- matrix(rnorm(n * m), n, m); colnames(Y) <- paste0("s", 1:m)
rownames(X) <- rownames(Y) <- paste0("g", 1:n)
X <- scale(X); Y <- scale(Y)

lambda <- 100; nrand <- 200

cat("\n--- RidgeFast (CPU, mt19937, ncores=1) ---\n")
r_fast <- RidgeFast::ridge(X, Y, lambda, nrand, ncores = 1L,
                           rng_method = "mt19937")

cat("\n--- RidgeCuda (GPU, mt19937) ---\n")
r_gpu  <- RidgeCuda::ridge(X, Y, lambda, nrand, ncores = 1L,
                           rng_method = "mt19937")

cat("\n=== PARITY (CPU-fast vs GPU, both mt19937) ===\n")
for (nm in c("beta", "se", "zscore", "pvalue")) {
  d <- max(abs(r_fast[[nm]] - r_gpu[[nm]]))
  status <- if (d < 1e-9) "PASS" else if (d < 1e-6) "NEAR" else "FAIL"
  cat(sprintf("  %-7s max|diff| = %.3e   [%s]\n", nm, d, status))
}

# Also compare against SecAct pure-R to close the loop
setwd("/vf/users/parks34/projects/1ridgesig/SecAct")
for (f in list.files("R", pattern="\\.R$", full.names=TRUE)) source(f)

r_pure <- .ridge_pureR(X, Y, lambda, nrand)

cat("\n=== PARITY (pure-R vs GPU, both mt19937) ===\n")
for (nm in c("beta", "se", "zscore", "pvalue")) {
  d <- max(abs(r_pure[[nm]] - r_gpu[[nm]]))
  status <- if (d < 1e-9) "PASS" else if (d < 1e-6) "NEAR" else "FAIL"
  cat(sprintf("  %-7s max|diff| = %.3e   [%s]\n", nm, d, status))
}

cat("\nDone.\n")
EOF
