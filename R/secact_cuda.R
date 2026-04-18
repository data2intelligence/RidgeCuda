#' Perform SecAct Inference using CUDA Ridge Regression
#'
#' This function performs ridge regression specifically tailored for SecAct
#' inference analysis, applying preprocessing steps consistent with common
#' workflows before running the CUDA-accelerated regression.
#'
#' @param expr_data A numeric matrix or data frame of expression data
#'   (genes x samples). Rownames should be gene identifiers.
#' @param sig_matrix Either a character string specifying a built-in signature
#'   matrix ("SecAct" or "CytoSig" - requires internal package data) or a
#'   numeric matrix/data frame (genes x features) providing a custom signature.
#'   Rownames should be gene identifiers matching `expr_data`.
#' @param lambda Ridge regularization parameter (lambda >= 0, default: 10000).
#'   Matches the default used in the Python example.
#' @param n_rand Number of permutations for significance testing (must be > 0,
#'   default: 1000).
#' @param batch_size Number of Y columns to process in each batch (default: 0, which means process
#'        all columns at once). Specify a smaller value to reduce memory usage for large datasets.
#' @param add_background Logical; if `TRUE`, compute the mean expression across
#'   samples for each gene in `expr_data` and add it as a 'background' feature
#'   to the signature matrix (default: `FALSE`).
#' @param scale_method Character string; method for scaling data before regression.
#'   Options are `"column"` (scale each column of X and Y independently to have
#'   mean 0 and standard deviation 1 using `base::scale()`) or `NULL` (no scaling).
#'   Default: `NULL`. Scaling is performed *before* adding background if both are enabled.
#' @param epsilon_scale Deprecated and ignored. Scaling uses `base::scale()`.
#' @param device_id Integer CUDA device ID to use (default: 0).
#' @return A list object of class `ridge_cuda` containing regression results,
#'   similar to the `ridge_cuda` function.
#' @examples
#' \dontrun{
#' # --- Requires internal data 'SecAct_signature_matrix' to be set up ---
#' # Create dummy internal data for example:
#' # gene_names <- paste0("Gene", 1:500)
#' # feature_names <- paste0("Feat", 1:50)
#' # sample_names <- paste0("Sample", 1:20)
#' # SecAct_signature_matrix <- matrix(rnorm(500*50), nrow=500, ncol=50,
#' #                                  dimnames=list(gene_names, feature_names))
#' # save(SecAct_signature_matrix, file = "R/sysdata.rda")
#' # expr_data <- matrix(rnorm(500*20), nrow=500, ncol=20,
#' #                     dimnames=list(gene_names, sample_names))
#' # --------------------------------------------------------------------
#'
#' # Basic run with internal SecAct signature (if data exists)
#' # secact_results <- secact_cuda(expr_data, sig_matrix = "SecAct")
#'
#' # Run with column scaling, background added, and batch processing
#' # secact_results_scaled <- secact_cuda(expr_data, sig_matrix = "SecAct",
#' #                                     add_background = TRUE, scale_method = "column",
#' #                                     batch_size = 5)
#'
#' # Run with a custom signature matrix
#' # custom_sig <- matrix(rnorm(100*10), nrow=100, ncol=10,
#' #                      dimnames=list(rownames(expr_data)[1:100], paste0("Feat", 1:10)))
#' # secact_results_custom <- secact_cuda(expr_data, sig_matrix = custom_sig, 
#' #                                     batch_size = 5)
#'
#' # print(secact_results_scaled)
#' # summary(secact_results_scaled)
#' }
#' @export
#' @importFrom methods is
#' @importFrom stats sd 
secact_cuda <- function(expr_data,
                        sig_matrix = "SecAct",
                        lambda = 10000,
                        n_rand = 1000,
                        batch_size = 0,
                        add_background = FALSE,
                        scale_method = NULL,
                        epsilon_scale = NULL, # Changed default, marked deprecated
                        device_id = 0) {

  # --- Argument Validation ---
  if (!is.matrix(expr_data) && !is.data.frame(expr_data)) {
    stop("'expr_data' must be a matrix or data frame.")
  }
  if (!is.numeric(as.matrix(expr_data))) {
    stop("'expr_data' must contain numeric values.")
  }
  if (is.null(rownames(expr_data))) {
    stop("'expr_data' must have rownames (gene identifiers).")
  }

  if (!is.character(sig_matrix) && !is.matrix(sig_matrix) && !is.data.frame(sig_matrix)) {
    stop("'sig_matrix' must be a character string ('SecAct', 'CytoSig'), matrix, or data frame.")
  }
  if (is.character(sig_matrix) && length(sig_matrix) == 1) {
     # Check if it's one of the recognized strings
     if (!sig_matrix %in% c("SecAct", "CytoSig")) {
         stop("If 'sig_matrix' is a string, it must be 'SecAct' or 'CytoSig'.")
     }
     # Loading logic implemented below
  } else if (is.matrix(sig_matrix) || is.data.frame(sig_matrix)) {
     if (!is.numeric(as.matrix(sig_matrix))) stop("'sig_matrix' must be numeric if provided as matrix/dataframe.")
     if (is.null(rownames(sig_matrix))) stop("'sig_matrix' must have rownames (gene identifiers).")
  }

  if (!is.null(scale_method) && !scale_method %in% c("column")) {
      stop("scale_method must be NULL or 'column'.")
  }
  
  # Validate batch_size
  if (!is.numeric(batch_size) || length(batch_size) != 1 || floor(batch_size) != batch_size) {
    stop("batch_size must be a single integer value.")
  }
  
   # Deprecation warning for epsilon_scale
   if (!is.null(epsilon_scale)) {
      warning("'epsilon_scale' is deprecated and ignored. Scaling uses base::scale().")
   }
  # Validation for lambda, n_rand, device_id is handled by ridge_cuda() call

  # --- Load/Prepare Signature Matrix (X) ---
  X_df_raw <- NULL
  if (is.character(sig_matrix) && length(sig_matrix) == 1) {
      message("Loading predefined signature matrix: ", sig_matrix)
      # Construct the expected object name
      sig_object_name <- paste0(sig_matrix, "_signature_matrix")
      # Attempt to load from the package's internal data environment
      internal_data_env <- new.env(parent = emptyenv())
      tryCatch({
          # This relies on the object being saved in R/sysdata.rda during build
          utils::data(list=sig_object_name, package = "RidgeRegCuda", envir = internal_data_env)
      }, error = function(e) {
          stop("Failed to load internal data '", sig_object_name,
               "' from package 'RidgeRegCuda'. Ensure it exists in 'R/sysdata.rda'. Error: ", e$message)
      })

      if (!exists(sig_object_name, envir = internal_data_env)) {
           stop("Internal data object '", sig_object_name, "' not found after loading. Check package build.")
      }
      X_df_raw <- get(sig_object_name, envir = internal_data_env)
      # Ensure it's a data frame for consistency
      if (!is.data.frame(X_df_raw)) X_df_raw <- as.data.frame(X_df_raw)

  } else { # User provided matrix/dataframe
      message("Using user-provided signature matrix.")
      X_df_raw <- as.data.frame(sig_matrix) # Ensure it's a dataframe
  }

  # --- Prepare Expression Data (Y) ---
  Y_df_raw <- as.data.frame(expr_data) # Ensure dataframe

  # --- Filter to Common Genes ---
  message("Preprocessing: Filtering to common genes...")
  # Ensure rownames are character type for matching
  rownames(Y_df_raw) <- as.character(rownames(Y_df_raw))
  rownames(X_df_raw) <- as.character(rownames(X_df_raw))

  common_genes <- intersect(rownames(Y_df_raw), rownames(X_df_raw))
  n_common <- length(common_genes)

  if (n_common == 0) {
    stop("No common genes found between expression data and signature matrix.")
  }
   message(sprintf("Found %d common genes.", n_common))

  Y_df <- Y_df_raw[common_genes, , drop = FALSE]
  X_df <- X_df_raw[common_genes, , drop = FALSE]
  message(sprintf("Data shapes after filtering: Y=(%s), X=(%s)",
                  paste(dim(Y_df), collapse=", "), paste(dim(X_df), collapse=", ")))

  # Clean up raw copies
  rm(Y_df_raw, X_df_raw)
  gc()

  # --- Convert to Matrix Before Scaling/Background ---
  X_mat <- as.matrix(X_df); rm(X_df); gc()
  Y_mat <- as.matrix(Y_df); rm(Y_df); gc()

  # --- Optional: Scaling (Applied Before Background) ---
  if (!is.null(scale_method)) {
      if (scale_method == "column") {
          message("Preprocessing: Performing COLUMN-WISE scaling on Y and X...")
          # scale() returns a matrix
          X_mat <- scale(X_mat, center = TRUE, scale = TRUE)
          Y_mat <- scale(Y_mat, center = TRUE, scale = TRUE)

          # Handle potential NaN/Inf from scaling columns with zero std dev
          X_mat[!is.finite(X_mat)] <- 0.0
          Y_mat[!is.finite(Y_mat)] <- 0.0
          message("Column-wise scaling complete.")
      }
      # Add 'global' scaling here if needed
  } else {
      message("Preprocessing: No internal scaling applied.")
  }

  # --- Optional: Add Background (Applied After Scaling) ---
  X_final <- X_mat # Start with potentially scaled X
  if (add_background) {
    message("Preprocessing: Adding background column to X (after potential scaling)...")
    if ("background" %in% colnames(X_final)) {
      warning("Signature matrix already has a 'background' column. Overwriting.")
    }
    # Compute background from potentially scaled Y
    background_col <- rowMeans(Y_mat, na.rm = TRUE)
    # Add as a new column to the potentially scaled X
    X_final <- cbind(X_final, background = background_col)
    message(sprintf("Shape of X after adding background: (%s)", paste(dim(X_final), collapse=", ")))
  }
  Y_final <- Y_mat # Use the potentially scaled Y

  # --- Ensure Final Matrices are Double Precision ---
  storage.mode(X_final) <- "double"
  storage.mode(Y_final) <- "double"

  # --- Call Core Ridge Function ---
  message("Calling ridge_cuda for computation...")
  # Store the call for the result object, manually constructing it
  call_obj <- match.call()

  # The main ridge_cuda function handles Y type check and conversion if needed
  result <- ridge_cuda(X = X_final,
                       Y = Y_final, # Pass the potentially scaled matrix
                       lambda = lambda,
                       n_rand = n_rand,
                       batch_size = batch_size,
                       device_id = device_id)

  # Override the call stored by ridge_cuda with the call to secact_cuda
  result$call <- call_obj
  message("secact_cuda inference finished.")

  return(result)
}