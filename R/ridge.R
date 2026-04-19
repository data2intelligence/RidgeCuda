#' GPU-accelerated ridge regression with permutation testing
#'
#' Adapter matching the shared accelerator API used by
#' \code{RidgeFast::ridge}. Dispatches to the CUDA kernel
#' via \code{\link{ridge_cuda}} (or the \code{_with_perm_r} entry point
#' for canonical MT19937 mode).
#'
#' @section Reproducibility:
#' With \code{rng_method = "mt19937"}, an MT19937 (seed 0) permutation
#' table is generated on the host and uploaded to the GPU. This gives
#' the same permutation sequence as \code{RidgeFast} and SecAct's
#' pure-R backend, so output is bit-identical up to floating-point
#' accumulation order. With \code{rng_method = "srand"} the kernel
#' uses an in-process Fisher-Yates with C stdlib rand() — faster, but
#' not reproducible.
#'
#' @param X Numeric matrix, n x p, column-scaled signature matrix.
#' @param Y Numeric matrix, n x m, column-scaled expression matrix.
#' @param lambda Ridge penalty (default 5e+05).
#' @param nrand Number of permutations (default 1000).
#' @param ncores Ignored on GPU (GPU parallelism is intrinsic).
#' @param rng_method \code{"mt19937"} (default) or \code{"srand"}.
#' @param device_id CUDA device index (default 0).
#' @return A list with four p-by-m matrices: \code{beta}, \code{se},
#'   \code{zscore}, \code{pvalue}.
#' @seealso \code{\link{ridge_cuda}}
#' @useDynLib RidgeCuda, .registration = TRUE
#' @export
ridge <- function(X, Y, lambda = 5e+05, nrand = 1000L,
                  ncores = 1L, rng_method = "mt19937",
                  device_id = 0L) {
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.matrix(Y) && !inherits(Y, "Matrix")) Y <- as.matrix(Y)
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

  use_canonical <- rng_norm == "mt19937" && is.matrix(Y)
  if (use_canonical) {
    storage.mode(Y) <- "double"
    cuda_status <- check_cuda_available(device_id = as.integer(device_id))
    if (!cuda_status$available) {
      stop("CUDA initialization failed: ", cuda_status$message)
    }
    # CUDA's permuteColumnsKernel reads T[:, indices[j]]; RidgeFast's
    # Tcol kernel reads T[:, inv[j]]. Invert so both produce the same
    # permuted T, which is mathematically a row-permutation of Y.
    n <- nrow(X)
    fwd_table <- .gsl_mt19937_perm_table(n, as.integer(nrand))
    inv_table <- matrix(0L, nrow = nrow(fwd_table), ncol = ncol(fwd_table))
    col_ids_0 <- seq_len(n) - 1L
    for (r in seq_len(nrow(fwd_table))) {
      inv_table[r, fwd_table[r, ] + 1L] <- col_ids_0
    }
    storage.mode(inv_table) <- "integer"
    res <- .Call("ridge_cuda_dense_with_perm_r",
                 X, Y, as.double(lambda), as.integer(nrand),
                 0L, as.integer(device_id), inv_table,
                 PACKAGE = "RidgeCuda")
  } else {
    res <- ridge_cuda(X = X, Y = Y,
                      lambda = lambda,
                      n_rand = as.integer(nrand),
                      batch_size = 0L,
                      device_id = as.integer(device_id))
  }
  wrap(res)
}
