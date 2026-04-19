#' GPU-accelerated ridge regression with permutation testing
#'
#' Adapter matching the shared accelerator API used by
#' \code{RidgeRegFast::Ridge.Reg.Fast}. Calls the CUDA kernel via
#' \code{\link{ridge_cuda}} under the hood.
#'
#' @section Reproducibility:
#' The current CUDA kernel uses cuRAND for in-GPU permutation, which
#' does NOT match the MT19937 sequence used by RidgeRegFast and
#' SecAct's pure-R backend. Consequently, results from RidgeRegCuda
#' are \emph{statistically equivalent} to the CPU backends but
#' \emph{not bit-identical}. For bitwise cross-backend reproducibility
#' use \code{RidgeRegFast::Ridge.Reg.Fast} with \code{ncores = 1}.
#'
#' Future work: add an entry point that accepts an externally-generated
#' MT19937 permutation table, enabling GPU canonical-mode parity.
#'
#' @param X Numeric matrix, n x p, column-scaled signature matrix.
#' @param Y Numeric matrix, n x m, column-scaled expression matrix.
#' @param lambda Ridge penalty (default 5e+05).
#' @param nrand Number of permutations (default 1000).
#' @param ncores Ignored on GPU (GPU parallelism is intrinsic). Present
#'   only to match the shared accelerator API.
#' @param rng_method Accepted values: \code{"mt19937"} (accepted with a
#'   parity warning), \code{"srand"}, or \code{"curand"}. All map to
#'   cuRAND internally at the moment.
#' @param device_id CUDA device index (default 0).
#' @return A list with four p-by-m matrices: \code{beta}, \code{se},
#'   \code{zscore}, \code{pvalue}.
#' @seealso \code{\link{ridge_cuda}}, \code{\link{secact_cuda}}
#' @export
Ridge.Reg.Fast <- function(X, Y, lambda = 5e+05, nrand = 1000L,
                           ncores = 1L, rng_method = "mt19937",
                           device_id = 0L) {
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.matrix(Y)) Y <- as.matrix(Y)
  if (!is.numeric(X) || !is.numeric(Y)) stop("X and Y must be numeric.")
  if (nrow(X) != nrow(Y)) stop("nrow(X) must equal nrow(Y).")

  rng_norm <- tolower(rng_method)
  if (!rng_norm %in% c("mt19937", "gsl", "srand", "curand")) {
    stop("rng_method must be one of 'mt19937', 'gsl', 'srand', 'curand'.")
  }
  if (rng_norm %in% c("mt19937", "gsl")) {
    warning(
      "RidgeRegCuda uses cuRAND internally; results will not be ",
      "bit-identical to RidgeRegFast/SecAct pure-R (both use MT19937). ",
      "For bitwise reproducibility use RidgeRegFast with ncores=1."
    )
  }

  storage.mode(X) <- "double"
  storage.mode(Y) <- "double"

  res <- ridge_cuda(X = X, Y = Y,
                   lambda = lambda,
                   n_rand = as.integer(nrand),
                   batch_size = 0L,
                   device_id = as.integer(device_id))

  p <- ncol(X); m <- ncol(Y)
  dn <- list(colnames(X), colnames(Y))
  pick <- function(nm) {
    v <- res[[nm]]
    if (is.matrix(v)) {
      if (!identical(dim(v), c(p, m))) {
        v <- matrix(as.numeric(v), p, m, byrow = TRUE)
      }
    } else {
      v <- matrix(as.numeric(v), p, m, byrow = TRUE)
    }
    dimnames(v) <- dn
    v
  }
  list(
    beta   = pick("beta"),
    se     = pick("se"),
    zscore = pick("zscore"),
    pvalue = pick("pvalue")
  )
}
