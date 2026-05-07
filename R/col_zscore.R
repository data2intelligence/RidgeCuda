#' In-place column z-score for a numeric matrix
#'
#' Subtracts the column mean and divides by the column sample standard
#' deviation (\code{sd} with \code{ddof = 1}, matching
#' \code{matrixStats::colSds}), modifying the input matrix in place.
#' Columns with zero standard deviation are scaled by 1 (left unchanged
#' relative to the mean shift) — same convention as
#' \code{matrixStats::colSds(Y)} followed by \code{sd[sd == 0] <- 1}.
#'
#' @section Why this exists:
#' R's copy-on-modify semantics make the natural R-level idiom
#' \preformatted{for (j in seq_along(mu)) Y[, j] <- (Y[, j] - mu[j]) / sigma[j]}
#' allocate a full duplicate of \code{Y} on the first column-write
#' whenever \code{Y}'s NAMED count is ≥ 1, which it always is inside
#' any function scope. For an \code{n × m} double matrix that's an
#' \code{8nm}-byte transient peak — at \code{n = 14k}, \code{m = 100k}
#' that's a ~11 GB host-RSS spike on top of the matrix itself.
#' \code{col_zscore_inplace} writes through the SEXP's \code{REAL}
#' pointer in C, bypassing R's copy machinery entirely. RSS impact
#' on a 14k × 100k double matrix dropped from ~41 GB to ~25 GB in
#' our benchmarks (close to numpy's in-place behaviour).
#'
#' @section Safety:
#' MUTATES \code{Y} in place. Only call when you know \code{Y} isn't
#' aliased — typical safe pattern is right after loading from disk:
#' \preformatted{Y <- h5read("file.h5", "Y"); Y <- col_zscore_inplace(Y)}
#' Calling on a matrix that's also referenced elsewhere will
#' silently mutate the other reference. The return value is the
#' same SEXP — capture it (\code{Y <- col_zscore_inplace(Y)}) so
#' downstream code uses the modified matrix.
#'
#' @section Why no \code{storage.mode<-} coercion:
#' This wrapper avoids any R-level assignment that would trigger
#' \code{duplicate()} (e.g. \code{storage.mode(Y) <- "double"}). The
#' caller is responsible for passing a \code{REALSXP} matrix; the
#' C entry point checks via \code{isReal()} and errors otherwise.
#' Without this discipline the wrapper would silently shadow Y
#' with a copy and the in-place modification would never reach the
#' caller's binding (which is exactly the bug discovered during
#' integration testing — the assertion fired with
#' \code{max|diff| = 9.0} because Y was untouched).
#'
#' @param Y A numeric \code{matrix} (n × m, n ≥ 2,
#'   \code{storage.mode(Y) == "double"}). Modified in place.
#' @return \code{Y}, modified in place. Same SEXP as the input.
#' @export
col_zscore_inplace <- function(Y) {
    if (!is.matrix(Y))
        stop("Y must be a matrix.")
    if (storage.mode(Y) != "double")
        stop("Y must be storage.mode='double' (use storage.mode(Y)<-'double' ",
             "BEFORE calling this — a coercion inside the wrapper would ",
             "trigger an R-level duplicate that defeats the in-place modify).")
    invisible(.Call("col_zscore_inplace_r", Y, PACKAGE = "RidgeCuda"))
}
