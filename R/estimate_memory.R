#' Estimate GPU Memory Requirements
#'
#' Estimates the GPU memory (in bytes) needed for a `ridge_cuda` computation.
#'
#' @param n_genes Number of genes (rows).
#' @param n_features Number of features (columns in X).
#' @param n_samples Number of samples (columns in Y).
#' @param n_rand Number of permutations (must be > 0).
#' @param nnz Approximate number of non-zero elements if Y is sparse.
#'            Only used if `is_sparse = TRUE`.
#' @param is_sparse Logical indicating if Y is sparse (`TRUE`) or dense (`FALSE`).
#' @param batch_size Number of Y columns to process in each batch (0 means process all at once).
#'                  Specifying a smaller batch size will reduce memory usage.
#' @param device_id CUDA device ID (default: 0). Used only to ensure CUDA is initialized.
#' @return A list containing input parameters and estimated `required_bytes`.
#' @examples
#' \dontrun{
#' # Estimate for a dense calculation with all columns at once
#' mem_dense_full <- estimate_cuda_memory(n_genes=10000, n_features=100, n_samples=50, n_rand=1000)
#' print(paste("Dense (all columns): Requires approx:", round(mem_dense_full$required_bytes / 1024^2), "MB"))
#'
#' # Estimate for a dense calculation with batched processing
#' mem_dense_batch <- estimate_cuda_memory(n_genes=10000, n_features=100, n_samples=50,
#'                                        n_rand=1000, batch_size=10)
#' print(paste("Dense (10 cols/batch): Requires approx:", round(mem_dense_batch$required_bytes / 1024^2), "MB"))
#'
#' # Estimate for a sparse calculation (assuming 5% density) with all columns at once
#' nnz_est <- ceiling(10000 * 50 * 0.05)
#' mem_sparse_full <- estimate_cuda_memory(n_genes=10000, n_features=100, n_samples=50,
#'                                        n_rand=1000, nnz=nnz_est, is_sparse=TRUE)
#' print(paste("Sparse (all columns): Requires approx:", round(mem_sparse_full$required_bytes / 1024^2), "MB"))
#'
#' # Estimate for a sparse calculation with batched processing
#' mem_sparse_batch <- estimate_cuda_memory(n_genes=10000, n_features=100, n_samples=50,
#'                                         n_rand=1000, nnz=nnz_est, is_sparse=TRUE, batch_size=10)
#' print(paste("Sparse (10 cols/batch): Requires approx:", round(mem_sparse_batch$required_bytes / 1024^2), "MB"))
#' }
#' @export
#' @useDynLib RidgeCuda, .registration = TRUE
estimate_cuda_memory <- function(n_genes, n_features, n_samples, n_rand = 1000,
                               nnz = NULL, is_sparse = FALSE, batch_size = 0, device_id = 0) {

  # Validate inputs
  if (!is.numeric(n_genes) || length(n_genes)!=1 || n_genes <= 0 || floor(n_genes) != n_genes) {
      stop("n_genes must be positive integer.")
  }
  if (!is.numeric(n_features) || length(n_features)!=1 || n_features <= 0 || floor(n_features) != n_features) {
      stop("n_features must be positive integer.")
  }
  if (!is.numeric(n_samples) || length(n_samples)!=1 || n_samples <= 0 || floor(n_samples) != n_samples) {
      stop("n_samples must be positive integer.")
  }
  # Permutation only, n_rand must be positive integer
  if (!is.numeric(n_rand) || length(n_rand)!=1 || n_rand <= 0 || floor(n_rand) != n_rand) {
      stop("n_rand must be a positive integer.")
  }
  if (!is.logical(is_sparse) || length(is_sparse)!=1) stop("is_sparse must be a single logical value.")
  if (!is.numeric(device_id) || length(device_id) != 1 || device_id < 0 || floor(device_id) != device_id) {
    stop("device_id must be a single non-negative integer.")
  }
  if (!is.numeric(batch_size) || length(batch_size) != 1 || floor(batch_size) != batch_size) {
    stop("batch_size must be a single integer value.")
  }

  nnz_val <- if (is_sparse && !is.null(nnz)) {
      if(!is.numeric(nnz) || length(nnz)!=1 || nnz < 0 || floor(nnz) != nnz) {
          stop("nnz must be non-negative integer when provided for sparse input.")
      }
      as.integer(nnz)
  } else if (is_sparse && is.null(nnz)) {
      warning("nnz not provided for sparse calculation; using placeholder 0. Estimation may be inaccurate.")
      0L
  } else {
      0L # Not used for dense
  }

  # Ensure CUDA is initialized (silently checks/initializes)
  init_stat <- check_cuda_available(device_id = as.integer(device_id))
  if (init_stat$status != 0) {
      stop("CUDA initialization failed, cannot estimate memory. Message: ", init_stat$message)
  }

  # Call C++ function registered as "ridge_cuda_memory_requirements_r".
  # `is_sparse` is passed through as a LOGICAL because the C side calls
  # `isLogical()` on it before `asLogical()`. An earlier as.integer()
  # wrapper mismatched this check and surfaced as:
  #   "is_sparse must be single logical"
  # whenever ridge_cuda() was reached via the non-canonical code path
  # (e.g. rng_method="srand" in RidgeCuda::ridge).
  result <- .Call("ridge_cuda_memory_requirements_r",
                 as.integer(n_genes), as.integer(n_features), as.integer(n_samples),
                 nnz_val, is_sparse, as.integer(n_rand), as.integer(batch_size),
                 PACKAGE = "RidgeCuda")

  # Convert output bytes to numeric for easier use in R
  result$required_bytes <- as.numeric(result$required_bytes)
  # Ensure list elements have correct types before returning
  result$n_genes <- as.integer(n_genes)
  result$n_features <- as.integer(n_features)
  result$n_samples <- as.integer(n_samples)
  result$n_rand <- as.integer(n_rand)
  result$is_sparse <- as.logical(is_sparse) # Keep as logical in R output
  result$batch_size <- as.integer(batch_size)

  return(result)
}