#' Batched GPU ridge regression
#'
#' Memory-efficient wrapper around \code{\link{ridge}} that processes
#' \code{Y} in column-batches on the GPU. Supports in-memory matrices,
#' HDF5 files, and user-supplied readers; results can be accumulated in
#' memory or streamed to an HDF5 output file.
#'
#' The observed \eqn{\beta} and permutation statistics are independent
#' across columns of \code{Y}, so batched output is bit-identical (up to
#' floating-point accumulation order) to a single \code{ridge()} call.
#' The MT19937 permutation table and the \eqn{T} projection are rebuilt
#' per batch in v1; context-based reuse is a v2 optimization.
#'
#' @param X Numeric matrix, n x p, column-scaled signature matrix.
#' @param Y Either (a) a dense matrix n x m, (b) a path to an HDF5 file
#'   containing a dataset named \code{"Y"} of shape n x m (requires the
#'   \pkg{rhdf5} package), or (c) \code{NULL} together with a
#'   \code{reader} callback.
#' @param lambda Ridge penalty (default 5e+05).
#' @param nrand Number of permutations (default 1000).
#' @param ncores Ignored on GPU (kept for API symmetry).
#' @param rng_method \code{"mt19937"} (default) or \code{"srand"}.
#' @param device_id CUDA device index (default 0).
#' @param batch_size Number of Y-columns per batch (default 5000).
#' @param reader Optional \code{function(start, end)} returning the
#'   n x (end - start + 1) slice of \code{Y}. Requires \code{n_samples}.
#' @param n_samples Total number of samples (columns of \code{Y});
#'   required when \code{reader} is used.
#' @param output_h5 Optional path to an HDF5 file; if supplied the four
#'   p x m result matrices are streamed to datasets
#'   \code{"beta" / "se" / "zscore" / "pvalue"} and the function returns
#'   metadata instead of the matrices (requires \pkg{rhdf5}).
#' @param verbose Logical; print per-batch progress messages.
#' @return If \code{output_h5} is \code{NULL}: a list with four
#'   p x m matrices. Otherwise a list with metadata about the written
#'   file (\code{path}, \code{p}, \code{m}, \code{num_batches}).
#' @seealso \code{\link{ridge}}
#' @export
ridge_batch <- function(X, Y, lambda = 5e+05, nrand = 1000L,
                        ncores = 1L, rng_method = "mt19937",
                        device_id = 0L,
                        batch_size = 5000L,
                        reader = NULL, n_samples = NULL,
                        output_h5 = NULL, verbose = FALSE) {
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.numeric(X)) stop("X must be numeric.")
  storage.mode(X) <- "double"
  n <- nrow(X); p <- ncol(X)
  sig_names <- colnames(X)
  if (is.null(sig_names)) sig_names <- paste0("sig", seq_len(p))

  batch_size <- as.integer(batch_size)
  if (is.na(batch_size) || batch_size < 1L) stop("batch_size must be >= 1.")

  src <- .resolve_y_source(Y, reader, n_samples, n)
  m <- src$m
  samp_names <- src$samp_names
  reader_fn <- src$reader_fn

  h5_out <- !is.null(output_h5)
  if (h5_out) {
    if (!requireNamespace("rhdf5", quietly = TRUE))
      stop("'output_h5' requires the 'rhdf5' package.")
    if (file.exists(output_h5)) file.remove(output_h5)
    rhdf5::h5createFile(output_h5)
    chunk_m <- min(batch_size, m)
    for (nm in c("beta", "se", "zscore", "pvalue")) {
      rhdf5::h5createDataset(output_h5, nm, dims = c(p, m),
                             storage.mode = "double",
                             chunk = c(p, chunk_m))
    }
    rhdf5::h5write(sig_names, output_h5, "signature_names")
    rhdf5::h5write(samp_names, output_h5, "sample_names")
  } else {
    out_beta   <- matrix(0, p, m, dimnames = list(sig_names, samp_names))
    out_se     <- matrix(0, p, m, dimnames = list(sig_names, samp_names))
    out_zscore <- matrix(0, p, m, dimnames = list(sig_names, samp_names))
    out_pvalue <- matrix(0, p, m, dimnames = list(sig_names, samp_names))
  }

  num_batches <- as.integer(ceiling(m / batch_size))
  for (b in seq_len(num_batches)) {
    s <- (b - 1L) * batch_size + 1L
    e <- min(as.integer(b * batch_size), m)
    if (verbose) message(sprintf("[ridge_batch/GPU] batch %d/%d (cols %d-%d)", b, num_batches, s, e))

    Y_batch <- reader_fn(s, e)
    if (!is.matrix(Y_batch)) Y_batch <- as.matrix(Y_batch)
    storage.mode(Y_batch) <- "double"
    if (nrow(Y_batch) != n)
      stop(sprintf("reader returned %d rows, expected %d.", nrow(Y_batch), n))
    if (ncol(Y_batch) != (e - s + 1L))
      stop(sprintf("reader returned %d cols for batch (%d-%d), expected %d.",
                   ncol(Y_batch), s, e, e - s + 1L))
    if (is.null(colnames(Y_batch))) colnames(Y_batch) <- samp_names[s:e]

    res <- ridge(X, Y_batch, lambda = lambda, nrand = nrand,
                 ncores = ncores, rng_method = rng_method,
                 device_id = device_id)

    if (h5_out) {
      for (nm in c("beta", "se", "zscore", "pvalue")) {
        rhdf5::h5write(res[[nm]], output_h5, nm, index = list(NULL, s:e))
      }
    } else {
      out_beta[, s:e]   <- res$beta
      out_se[, s:e]     <- res$se
      out_zscore[, s:e] <- res$zscore
      out_pvalue[, s:e] <- res$pvalue
    }
  }

  if (h5_out) {
    rhdf5::h5closeAll()
    invisible(list(path = output_h5, p = p, m = m, num_batches = num_batches))
  } else {
    list(beta = out_beta, se = out_se, zscore = out_zscore, pvalue = out_pvalue)
  }
}

.resolve_y_source <- function(Y, reader, n_samples, n) {
  if (is.character(Y) && length(Y) == 1L) {
    if (!requireNamespace("rhdf5", quietly = TRUE))
      stop("Reading Y from an HDF5 file requires the 'rhdf5' package.")
    info <- rhdf5::h5ls(Y)
    y_row <- info[info$name == "Y", ]
    if (nrow(y_row) == 0L) stop("HDF5 file '", Y, "' must contain a dataset named 'Y'.")
    dims <- as.integer(strsplit(y_row$dim[1], " x ", fixed = TRUE)[[1]])
    if (length(dims) != 2L) stop("'Y' dataset must be 2-dimensional.")
    if (dims[1] != n) stop(sprintf("HDF5 Y rows (%d) do not match nrow(X) (%d).", dims[1], n))
    m <- dims[2]
    samp_names <- if ("sample_names" %in% info$name) {
      as.character(rhdf5::h5read(Y, "sample_names"))
    } else paste0("s", seq_len(m))
    path <- Y
    list(m = m, samp_names = samp_names,
         reader_fn = function(s, e) {
           rhdf5::h5read(path, "Y", index = list(NULL, s:e))
         })
  } else if (!is.null(reader)) {
    if (is.null(n_samples)) stop("'reader' requires 'n_samples'.")
    m <- as.integer(n_samples)
    list(m = m, samp_names = paste0("s", seq_len(m)), reader_fn = reader)
  } else {
    if (!is.matrix(Y)) Y <- as.matrix(Y)
    if (nrow(Y) != n) stop(sprintf("nrow(Y) (%d) != nrow(X) (%d).", nrow(Y), n))
    m <- ncol(Y)
    samp_names <- if (is.null(colnames(Y))) paste0("s", seq_len(m)) else colnames(Y)
    list(m = m, samp_names = samp_names,
         reader_fn = function(s, e) Y[, s:e, drop = FALSE])
  }
}
