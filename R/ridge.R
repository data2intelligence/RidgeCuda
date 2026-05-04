#' GPU-accelerated ridge regression with permutation testing
#'
#' Adapter matching the shared accelerator API used by
#' \code{RidgeFast::ridge}. For dense Y, generates a permutation table
#' on the host using the requested RNG (\code{srand} or
#' \code{mt19937}), then dispatches to the CUDA kernel via
#' \code{\link[=ridge_cuda_dense_with_perm_r]{ridge_cuda_dense_with_perm_r}}.
#' Sparse Y falls back to \code{\link{ridge_cuda}}.
#'
#' @section Reproducibility:
#' With \code{rng_method = "srand"} (default), a host-side
#' Fisher-Yates with C stdlib rand() seeded by \code{seed} produces
#' the same permutation stream as \code{RidgeFast::ridge(..., rng_method =
#' "srand")} and SecAct's original C backend — backward-compat with
#' published results. With \code{rng_method = "mt19937"}, the GSL
#' MT19937 table is generated on host via pure R and uploaded to the
#' GPU; bit-identical (up to floating-point accumulation order) with
#' RidgeFast and SecAct's pure-R backend under the same seed.
#'
#' @param X Numeric matrix, n x p, column-scaled signature matrix.
#' @param Y Numeric matrix, n x m, column-scaled expression matrix.
#' @param lambda Ridge penalty (default 5e+05).
#' @param nrand Number of permutations (default 1000).
#' @param ncores Ignored on GPU (GPU parallelism is intrinsic).
#' @param rng_method \code{"srand"} (default, matches original SecAct
#'   C behavior — platform-dependent stream) or \code{"mt19937"}
#'   (GSL MT19937, cross-platform reproducible).
#' @param seed Integer seed for the RNG (default 0).
#' @param device_id CUDA device index (default 0).
#' @return A list with four p-by-m matrices: \code{beta}, \code{se},
#'   \code{zscore}, \code{pvalue}.
#' @seealso \code{\link{ridge_cuda}}
#' @useDynLib RidgeCuda, .registration = TRUE
#' @export
ridge <- function(X, Y, lambda = 5e+05, nrand = 1000L,
                  ncores = 1L, rng_method = "srand", seed = 0L,
                  device_id = 0L) {
  if (!is.matrix(X)) X <- as.matrix(X)
  # Y may be a dense matrix or a dgCMatrix (CSC sparse). The two cases
  # route to different CUDA kernels:
  #   dense    → ridge_cuda_dense_with_perm_r   (cublasDgemm path)
  #   sparse   → ridge_cuda_sparse_with_perm_r  (cusparseSpMM path)
  # Both consume the same caller-supplied perm table for cross-backend
  # bitwise reproducibility. No host-side densify of dgCMatrix anymore —
  # that was a temporary "API parity" fallback before the perm-aware
  # sparse path landed.
  Y_is_sparse <- inherits(Y, "dgCMatrix") || inherits(Y, "CsparseMatrix") ||
                 inherits(Y, "Matrix")
  if (!Y_is_sparse && !is.matrix(Y)) Y <- as.matrix(Y)
  if (Y_is_sparse) {
    if (!inherits(Y, "dgCMatrix")) Y <- methods::as(Y, "CsparseMatrix")
    if (!inherits(Y, "dgCMatrix")) stop("Sparse Y must coerce to dgCMatrix.")
  } else if (!is.numeric(Y)) {
    stop("Y must be numeric (dense matrix or dgCMatrix).")
  }
  if (!is.numeric(X)) stop("X must be numeric.")
  if (nrow(X) != nrow(Y)) stop("nrow(X) must equal nrow(Y).")

  rng_norm <- match.arg(tolower(rng_method), c("mt19937", "srand"))
  storage.mode(X) <- "double"
  p <- ncol(X); m <- ncol(Y)
  dn <- list(colnames(X), colnames(Y))

  wrap <- function(res) {
    out <- lapply(res[c("beta", "se", "zscore", "pvalue")], function(v) {
      if (!is.matrix(v) || !identical(dim(v), c(p, m))) {
        v <- matrix(as.numeric(v), p, m, byrow = TRUE)
      }
      dimnames(v) <- dn
      v
    })
    names(out) <- c("beta", "se", "zscore", "pvalue")
    out
  }

  if (!Y_is_sparse) storage.mode(Y) <- "double"
  cuda_status <- check_cuda_available(device_id = as.integer(device_id))
  if (!cuda_status$available) {
    stop("CUDA initialization failed: ", cuda_status$message)
  }
  n <- nrow(X)
  # Build forward perm table on host per requested RNG.
  fwd_table <- if (rng_norm == "mt19937") {
    .gsl_mt19937_perm_table(n, as.integer(nrand), seed = as.integer(seed))
  } else {
    .Call("build_srand_perm_table_r",
          as.integer(n), as.integer(nrand), as.integer(seed),
          PACKAGE = "RidgeCuda")
  }
  # CUDA's permuteColumnsKernel reads T[:, indices[j]]; RidgeFast's
  # Tcol kernel reads T[:, inv[j]]. Invert so both produce the same
  # permuted T, which is mathematically a row-permutation of Y.
  inv_table <- matrix(0L, nrow = nrow(fwd_table), ncol = ncol(fwd_table))
  col_ids_0 <- seq_len(n) - 1L
  for (r in seq_len(nrow(fwd_table))) {
    inv_table[r, fwd_table[r, ] + 1L] <- col_ids_0
  }
  storage.mode(inv_table) <- "integer"
  res <- if (Y_is_sparse) {
    .Call("ridge_cuda_sparse_with_perm_r",
          X, Y, as.double(lambda), as.integer(nrand),
          0L, as.integer(device_id), inv_table,
          PACKAGE = "RidgeCuda")
  } else {
    .Call("ridge_cuda_dense_with_perm_r",
          X, Y, as.double(lambda), as.integer(nrand),
          0L, as.integer(device_id), inv_table,
          PACKAGE = "RidgeCuda")
  }
  wrap(res)
}
