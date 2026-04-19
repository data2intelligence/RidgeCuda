# Check the current directory structure and add proper export tags

# 1. Make sure the ridge_cuda.R file includes proper export tags for the scaling functions
# Find where these functions are defined (likely in ridge_cuda.R) and ensure they have the @export tag

#' @title Scale Dense Matrix Columns using CUDA
#' @description Performs column-wise standard scaling (Z-score) on a dense matrix
#'   using CUDA acceleration.
#'
#' @param mat A numeric matrix.
#' @param device_id The CUDA device ID to use (default: 0).
#'
#' @return A list containing:
#'   \item{scaled_matrix}{The column-scaled matrix with original dimnames.}
#'   \item{center}{A named numeric vector of the original column means.}
#'   \item{scale}{A named numeric vector of the original column standard deviations.}
#' @export
scale_dense_matrix_cuda <- function(mat, device_id = 0L) {
  if (!is.matrix(mat) || !is.numeric(mat)) {
    stop("Input 'mat' must be a numeric matrix.", call. = FALSE)
  }
  if (!is.integer(device_id) && !is.numeric(device_id)) {
     stop("device_id must be an integer.", call. = FALSE)
  }
  device_id <- as.integer(device_id)[1]

  # --- Check CUDA and Initialize ---
  cuda_status <- check_cuda_available(device_id = as.integer(device_id))
  if (!cuda_status$available) {
    stop("CUDA initialization failed for device ", device_id, ": ", cuda_status$message)
  }

  result <- .Call("ridge_cuda_scale_dense_matrix_r", mat, device_id)

  if (is.null(result)) {
    stop("CUDA dense scaling failed. Check logs or CUDA setup.", call. = FALSE)
  }
  # Attributes (like dimnames on vectors) should be preserved by C++ wrapper

  return(result)
}

#' @title Scale Sparse Matrix Columns (Sparse-Aware) using CUDA
#' @description Performs column-wise sparse-aware scaling on a dgCMatrix
#'   using CUDA acceleration. Calculates mean and standard deviation based
#'   only on non-zero elements per column and scales only those non-zero elements.
#'
#' @param mat A dgCMatrix object (from the Matrix package).
#' @param device_id The CUDA device ID to use (default: 0).
#'
#' @return A list containing:
#'   \item{scaled_matrix}{A new dgCMatrix object with scaled non-zero values.}
#'   \item{`scaled:center-nz`}{A named numeric vector of the means calculated from non-zeros per column.}
#'   \item{`scaled:scale-nz`}{A named numeric vector of the standard deviations calculated from non-zeros per column.}
#' @export
#' @importFrom Matrix sparseMatrix
scale_sparse_matrix_csc_cuda <- function(mat, device_id = 0L) {
   if (!inherits(mat, "dgCMatrix")) {
     stop("Input 'mat' must be a dgCMatrix object.", call. = FALSE)
   }
   if (!is.integer(device_id) && !is.numeric(device_id)) {
      stop("device_id must be an integer.", call. = FALSE)
   }
   device_id <- as.integer(device_id)[1]

   # --- Check CUDA and Initialize ---
   cuda_status <- check_cuda_available(device_id = as.integer(device_id))
   if (!cuda_status$available) {
     stop("CUDA initialization failed for device ", device_id, ": ", cuda_status$message)
   }

   result_list <- .Call("ridge_cuda_scale_sparse_matrix_csc_r", mat, device_id)

   if (is.null(result_list)) {
     stop("CUDA sparse scaling failed. Check logs or CUDA setup.", call. = FALSE)
   }

   # Reconstruct the dgCMatrix object in R
   # The .Call returns a list with components 'x', 'i', 'p', 'Dim' etc.
   reconstructed_mat <- tryCatch({
      Matrix::sparseMatrix(
          i = result_list$i,       # Row indices (0-based)
          p = result_list$p,       # Column pointers (0-based)
          x = result_list$x,       # Scaled non-zero values
          dims = result_list$Dim,  # Dimensions
          index1 = FALSE,          # Indicate 0-based indices
          dimnames = dimnames(mat) # Preserve original dimnames
        )
    }, error = function(e){
        warning("Failed to reconstruct sparse matrix: ", e$message, call. = FALSE)
        return(NULL)
    })

    if(is.null(reconstructed_mat)){
        warning("Returning raw list components instead of dgCMatrix object.", call. = FALSE)
        return(result_list)
    }

   # Return a list containing the reconstructed matrix and the scaling attributes
   final_result <- list(
        scaled_matrix = reconstructed_mat,
       `scaled:center-nz` = result_list$`scaled:center-nz`,
       `scaled:scale-nz` = result_list$`scaled:scale-nz`
   )

   return(final_result)
}

#' Ridge Regression with CUDA Acceleration (Permutation Test)
#'
#' Performs ridge regression using CUDA for acceleration. Supports dense or
#' sparse (dgCMatrix) input for Y and provides significance testing
#' via permutation test. T-test functionality is not included.
#'
#' @section Input Matrix Orientation:
#' Assumes input matrices `X` and `Y` (if dense) are in R's standard
#' column-major format. The underlying CUDA implementation expects this format.
#'
#' @section Batch Processing:
#' For large datasets that exceed GPU memory, the function can process `Y` columns
#' in smaller batches using the `batch_size` parameter. This reduces memory usage
#' at the expense of slightly longer computation time due to multiple passes. The
#' `X` matrix is loaded once and reused for all batches.
#'
#' @param X Input numeric matrix (n_genes x n_features).
#' @param Y Input numeric matrix or sparse matrix (dgCMatrix) (n_genes x n_samples).
#' @param lambda Ridge regularization parameter (lambda >= 0, default: 1.0).
#' @param n_rand Number of permutations for significance testing (must be > 0, default: 1000).
#' @param batch_size Number of Y columns to process in each batch (default: 0, which means process
#'        all columns at once). Specify a smaller value to reduce memory usage for large datasets.
#' @param device_id Integer CUDA device ID to use (default: 0).
#' @return A list object of class `ridge_cuda` containing regression results:
#'         \describe{
#'           \item{beta}{Matrix of beta coefficients (n_features x n_samples).}
#'           \item{se}{Matrix of standard errors derived from permutations.}
#'           \item{zscore}{Matrix of z-scores derived from permutations.}
#'           \item{pvalue}{Matrix of p-values derived from permutations.}
#'           \item{df}{Always `NA_real_` as only permutation tests are supported.}
#'           \item{status}{Integer status code from the CUDA backend (0 = success).}
#'           \item{call}{The matched call to the function.}
#'         }
#'         The list also includes a "message" attribute with status details.
#' @examples
#' \dontrun{
#' # Ensure CUDA is available
#' if (!check_cuda_available()$available) {
#'   stop("CUDA not available or initialization failed.")
#' }
#'
#' # --- Dense Example ---
#' n_g <- 500; n_f <- 50; n_s <- 10
#' X_dense <- matrix(rnorm(n_g * n_f), nrow = n_g, ncol = n_f)
#' Y_dense <- matrix(rnorm(n_g * n_s), nrow = n_g, ncol = n_s)
#' storage.mode(X_dense) <- "double"
#' storage.mode(Y_dense) <- "double"
#'
#' # Dense ridge regression with Permutation test - process all at once
#' result_dense_perm <- ridge_cuda(X_dense, Y_dense, lambda = 1.0, n_rand = 100)
#'
#' # Dense ridge regression with batch processing - 5 columns per batch
#' # Uses less memory by processing Y in smaller chunks
#' result_dense_batch <- ridge_cuda(X_dense, Y_dense, lambda = 1.0, n_rand = 100, batch_size = 5)
#'
#' # --- Sparse Example ---
#' if (requireNamespace("Matrix", quietly = TRUE)) {
#'   # Create a sparse Y matrix (e.g., 10% density)
#'   Y_sparse <- Matrix::rsparsematrix(n_g, n_s, density = 0.1)
#'   Y_sparse <- as(Y_sparse, "dgCMatrix") # Ensure CSC format
#'
#'   # Sparse ridge regression with batch processing - 5 columns per batch
#'   result_sparse_batch <- ridge_cuda(X_dense, Y_sparse, lambda = 1.0, 
#'                                     n_rand = 100, batch_size = 5)
#'   print(result_sparse_batch)
#'   summary(result_sparse_batch)
#' } else {
#'   print("Matrix package not installed. Skipping sparse example.")
#' }
#'
#' # Clean up CUDA resources when done
#' cleanup_cuda()
#' }
#' @export
#' @importFrom methods as   
#' @import Matrix
#' @useDynLib RidgeCuda, .registration = TRUE
ridge_cuda <- function(X, Y, lambda = 1.0, n_rand = 1000, batch_size = 0, device_id = 0) {

  # --- Input Validation ---
  if (!is.matrix(X) || !is.numeric(X)) {
    stop("X must be a numeric matrix.")
  }
   # Basic validation for Y type
  is_Y_dense <- is.matrix(Y) && is.numeric(Y)
  is_Y_sparse_Matrix <- inherits(Y, "Matrix") # Check inheritance from Matrix package

  if (!is_Y_dense && !is_Y_sparse_Matrix) {
      stop("Y must be a numeric matrix or a sparse matrix object inheriting from 'Matrix'.")
  }

  if (!is.numeric(lambda) || length(lambda) != 1 || lambda < 0) {
    stop("lambda must be a single non-negative numeric value.")
  }
  # Permutation only, n_rand must be positive integer
  if (!is.numeric(n_rand) || length(n_rand) != 1 || n_rand <= 0 || floor(n_rand) != n_rand) {
    stop("n_rand must be a single positive integer for permutation testing.")
  }
  # Validate batch_size
  if (!is.numeric(batch_size) || length(batch_size) != 1 || floor(batch_size) != batch_size) {
    stop("batch_size must be a single integer value.")
  }
  # Suggest appropriate batch_size if it's too large
  if (batch_size > ncol(Y)) {
    warning("batch_size (", batch_size, ") is larger than the number of Y columns (", 
            ncol(Y), "). Setting batch_size = 0 (process all columns at once).")
    batch_size <- 0
  }
  if (!is.numeric(device_id) || length(device_id) != 1 || device_id < 0 || floor(device_id) != device_id) {
    stop("device_id must be a single non-negative integer.")
  }

  # --- Check CUDA and Initialize ---
  cuda_status <- check_cuda_available(device_id = as.integer(device_id))
  if (!cuda_status$available) {
    stop("CUDA initialization failed for device ", device_id, ": ", cuda_status$message)
  }

  # --- Determine Sparse/Dense Path ---
  use_sparse_path <- is_Y_sparse_Matrix

  # --- Prepare Arguments and Call C++ ---
  result <- NULL
  lambda_val <- as.double(lambda)
  n_rand_val <- as.integer(n_rand)
  batch_size_val <- as.integer(batch_size)
  device_id_val <- as.integer(device_id)

  # Ensure storage mode is double (important for C++ interface)
  storage.mode(X) <- "double"
  if (is_Y_dense) storage.mode(Y) <- "double"

  # Determine if we should use batch processing based on memory availability
  if (batch_size_val <= 0) {
    # Estimate memory for full processing (no batching)
    mem_req <- estimate_cuda_memory(n_genes = nrow(X), n_features = ncol(X), 
                                   n_samples = ncol(Y), n_rand = n_rand_val,
                                   nnz = if(use_sparse_path) Matrix::nnzero(Y) else NULL,
                                   is_sparse = use_sparse_path, batch_size = 0)
    
    # Get available GPU memory
    gpu_mem <- get_cuda_memory_info()
    free_mem_mb <- gpu_mem$free_memory / (1024^2)
    req_mem_mb <- mem_req$required_bytes / (1024^2)
    
    # Check if memory needed exceeds available memory with a safety margin (80%)
    if (req_mem_mb > free_mem_mb * 0.8) {
      # Calculate a reasonable batch size that would fit comfortably
      # Aim to use at most 60% of free memory
      target_mem_mb <- free_mem_mb * 0.6
      
      # Estimate batch size 
      # Simple approach: scale down proportionally
      proposed_batch_size <- max(1, floor(ncol(Y) * target_mem_mb / req_mem_mb))
      
      # Ensure batch_size is reasonable (between 1 and 100, or ncol(Y) if smaller)
      proposed_batch_size <- min(proposed_batch_size, 100, ncol(Y))
      
      warning("Estimated memory requirement (", round(req_mem_mb), 
              " MB) exceeds 80% of available GPU memory (", round(free_mem_mb), 
              " MB). Automatically setting batch_size = ", proposed_batch_size, 
              " to reduce memory usage.")
      
      batch_size_val <- as.integer(proposed_batch_size)
    }
  }

  # Log the actual batch size being used if >0
  if (batch_size_val > 0) {
    message("Processing Y in batches of ", batch_size_val, " columns (", 
            ceiling(ncol(Y) / batch_size_val), " batches total).")
  }

  if (use_sparse_path) {
    if (!requireNamespace("Matrix", quietly = TRUE)) {
      # This check is slightly redundant due to earlier validation but good practice
      stop("Package 'Matrix' is required for sparse matrix support.")
    }
    if (!inherits(Y, "dgCMatrix")) {
      message("Converting sparse Y to dgCMatrix (CSC format).")
      Y <- as(Y, "dgCMatrix")
    }
    if (nrow(X) != nrow(Y)) stop("X and sparse Y must have the same number of rows (n_genes).")

    # Call the sparse C++ function (registered as "ridge_cuda_sparse_r")
    result <- .Call("ridge_cuda_sparse_r", X, Y, lambda_val,
                   n_rand_val, batch_size_val, device_id_val, PACKAGE = "RidgeCuda")

  } else { # Dense path
    if (!is_Y_dense) stop("Internal error: Y is not a dense matrix in the dense path.")
    if (nrow(X) != nrow(Y)) stop("X and dense Y must have the same number of rows (n_genes).")

    # Call the dense C++ function (registered as "ridge_cuda_dense_r")
    result <- .Call("ridge_cuda_dense_r", X, Y, lambda_val,
                   n_rand_val, batch_size_val, device_id_val, PACKAGE = "RidgeCuda")
  }

  # --- Process Results ---
  # Ensure df is NA as T-test was removed
  result$df <- NA_real_
  # Add class for S3 methods
  class(result) <- c("ridge_cuda", class(result))
  # Store the function call
  result$call <- match.call()
  # Check status and issue warning if non-zero
  msg <- attr(result, "message") # Retrieve message attribute added by C++ helper
  if (!is.null(result$status) && result$status != 0) {
      if (is.null(msg)) msg <- "Unknown error occurred in CUDA backend."
      warning(paste("ridge_cuda computation finished with non-zero status:", result$status, "-", msg))
  } else if (is.null(msg)){
      # If status is 0 but message is missing, add default success message attribute
      attr(result, "message") <- "Success"
  }

  return(result)
}