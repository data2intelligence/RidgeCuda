#' Print Method for ridge_cuda Objects
#'
#' @param x A `ridge_cuda` object.
#' @param ... Additional arguments (ignored).
#' @return Invisibly returns the input object `x`.
#' @method print ridge_cuda
#' @export
#' @keywords internal
print.ridge_cuda <- function(x, ...) {
  cat("Ridge Regression Results (RidgeCuda - Permutation Test)\n") # Updated title
  cat("--------------------------------------------------------\n")

  if (!is.null(x$call)) { cat("Call: "); print(x$call) }

  status_msg <- attr(x, "message")
  if (is.null(status_msg)) status_msg <- "No status message available."
  cat("Status:", x$status, "-", status_msg, "\n")

  if (!is.null(x$beta) && is.matrix(x$beta)) {
    p <- nrow(x$beta)
    m <- ncol(x$beta)
    cat("Dimensions: ", p, " features x ", m, " samples\n", sep = "")
  } else {
     cat("Dimensions: Result matrices not found or invalid.\n")
  }

  # Statistics Method - Always permutation now
  cat("Method: Permutation test\n")

  cat("--------------------------------------------------------\n")
  cat("Results stored in list components: beta, se, zscore, pvalue, status\n") # Removed df
  cat("Use str() or summary() for more details.\n")

  invisible(x)
}

#' Summary Method for ridge_cuda Objects
#'
#' @param object A `ridge_cuda` object.
#' @param ... Additional arguments (ignored).
#' @return A list object of class `summary.ridge_cuda`.
#' @method summary ridge_cuda
#' @export
#' @keywords internal
summary.ridge_cuda <- function(object, ...) {
  sum_obj <- list()
  sum_obj$status <- object$status
  sum_obj$message <- attr(object, "message")
  if(is.null(sum_obj$message)) sum_obj$message <- "N/A"
  sum_obj$call <- object$call

  # Method Info - Always permutation now
  sum_obj$method <- "Permutation test"
  sum_obj$df <- NA_real_ # Keep df as NA

  if (!is.null(object$beta) && is.matrix(object$beta)) {
    sum_obj$dimensions <- c(nrow(object$beta), ncol(object$beta))
    names(sum_obj$dimensions) <- c("features", "samples")
    beta_vec <- as.vector(object$beta); beta_vec <- beta_vec[!is.na(beta_vec)]
    if(length(beta_vec) > 0) sum_obj$beta_summary <- summary(beta_vec)
    else sum_obj$beta_summary <- "No valid beta coefficients found."

    if(!is.null(object$pvalue) && is.matrix(object$pvalue)){
        p_vec <- as.vector(object$pvalue); p_vec <- p_vec[!is.na(p_vec)]
        if(length(p_vec) > 0) {
            sum_obj$pvalue_summary <- summary(p_vec)
            thresholds <- c(0.1, 0.05, 0.01, 0.001)
            sig_counts <- sapply(thresholds, function(thr) sum(p_vec < thr))
            names(sig_counts) <- paste0("p < ", thresholds)
            sum_obj$significant_counts <- sig_counts
        } else {
            sum_obj$pvalue_summary <- "No valid p-values found."
            sum_obj$significant_counts <- NULL
        }
    } else {
        sum_obj$pvalue_summary <- "P-value matrix not found."
        sum_obj$significant_counts <- NULL
    }
  } else {
      sum_obj$dimensions <- c(NA_integer_, NA_integer_)
      names(sum_obj$dimensions) <- c("features", "samples")
      sum_obj$beta_summary <- "Beta matrix not found or invalid."
      sum_obj$pvalue_summary <- "P-value matrix not found or invalid."
      sum_obj$significant_counts <- NULL
  }

  class(sum_obj) <- "summary.ridge_cuda"
  return(sum_obj)
}

#' Print Method for summary.ridge_cuda Objects
#'
#' @param x A `summary.ridge_cuda` object.
#' @param ... Additional arguments (ignored).
#' @return Invisibly returns the input object `x`.
#' @method print summary.ridge_cuda
#' @export
#' @keywords internal
print.summary.ridge_cuda <- function(x, ...) {
  cat("Summary: Ridge Regression Results (RidgeCuda - Permutation Test)\n") # Updated title
  cat("==================================================================\n")

  if (!is.null(x$call)) { cat("Call:\n"); print(x$call); cat("\n") }
  cat("Status:", x$status, "-", x$message, "\n")
  cat("Method:", x$method, "\n")
  cat("Dimensions:", x$dimensions[1], "features x", x$dimensions[2], "samples\n")

  cat("\n--- Beta Coefficients Summary ---\n")
  if (is.numeric(x$beta_summary)) print(x$beta_summary)
  else cat(x$beta_summary, "\n")

  cat("\n--- P-value Summary ---\n")
  if (is.numeric(x$pvalue_summary)) print(x$pvalue_summary)
  else cat(x$pvalue_summary, "\n")

  if (!is.null(x$significant_counts)) {
    cat("\n--- Significant Results (uncorrected permutation p-values) ---\n") # Updated desc
    print(x$significant_counts)
  }

  cat("==================================================================\n")
  invisible(x)
}
