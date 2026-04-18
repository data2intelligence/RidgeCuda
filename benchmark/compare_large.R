#!/usr/bin/env Rscript

# Example of ridge regression using the RidgeRegCuda R package on a larger dataset,
# applying column-wise scaling (sparse-aware for Y, dense for X), comparing
# ONLY the beta matrix using Pearson correlation, saving the computed beta,
# and managing memory with batch processing.

# --- Load Required Libraries ---
suppressPackageStartupMessages({
  library(RidgeRegCuda)
  library(data.table)
  library(stats)
  library(utils)
  library(arrow)
})
library(tools)

rm(list=ls()); gc(); 

# --- Basic Logging ---
log_info <- function(...) { message(format(Sys.time(), "%Y-%m-%d %H:%M:%S - INFO - "), ...) }
log_warning <- function(...) { warning(format(Sys.time(), "%Y-%m-%d %H:%M:%S - WARNING - "), ..., call. = FALSE) }
log_error <- function(...) { stop(format(Sys.time(), "%Y-%m-%d %H:%M:%S - ERROR - "), ..., call. = FALSE) }
cat_print <- function(...) { cat(..., "\n") }

# --- Configuration ---
data_dir <- "/data/parks34/projects/RidgeInf/datasets"
if (!dir.exists(data_dir)) { log_error("Data directory not found: ", data_dir) }
signature_dir <- "/data/parks34/projects/RidgeInf/data/signature_matrices"
expr_path <- file.path(data_dir, "GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz")
sig_matrix_path <- file.path(signature_dir, "AllSigFilteredBy_MoranI_TCGA_ICGC_0.25_ds3.tsv.gz")
precomputed_dir <- "/data/parks34/projects/RidgeInf/ComparePy2R/r_secact_results"
if (!dir.exists(precomputed_dir)) { log_warning("Precomputed results directory not found: ", precomputed_dir) }
precomputed_filename <- "GSE131907_Lung_Cancer_normalized_log2TPM_matrix_R_beta.csv"
output_file <- "./computed_beta.csv"
LAMBDA_VAL <- 5e5
N_RAND_PERM <- 250
CUDA_DEVICE_ID <- 0
BETA_CORR_THRESHOLD <- 0.999

# --- Batch Processing Configuration ---
# Auto-select batch size or specify a fixed value
AUTO_BATCH_SIZE <- TRUE     # If TRUE, automatically determine batch size based on memory
FIXED_BATCH_SIZE <- 20      # Used only if AUTO_BATCH_SIZE is FALSE
MEMORY_USAGE_TARGET <- 0.7  # Target memory usage (70% of available GPU memory)

# --- CUDA Memory Options ---
ENABLE_MEMORY_POOLING <- TRUE
MEMORY_POOL_SIZE_MB <- 1024       # 1GB initial pool size
MEMORY_RELEASE_THRESHOLD_MB <- 256 # 256MB release threshold

# --- Helper Functions ---
load_beta_csv_result <- function(results_dir, base_filename) { # Changed base_filename name for clarity
  result <- list()
  title <- 'beta' # Only loading beta now

  # Construct the full path *before* tryCatch
  # Assuming base_filename is the full filename now as per your log message
  # f_name <- paste0(base_filename, "_", title, ".csv.gz") # Old way
  f_path <- file.path(results_dir, base_filename) # Use base_filename directly

  log_info(sprintf("Loading precomputed CSV result for '%s' from: %s", title, f_path))

  if (file.exists(f_path)) {
    tryCatch({
      dt <- data.table::fread(f_path, sep = ',', header = TRUE, check.names = FALSE)
      first_col_name <- names(dt)[1]

      # Check if first column looks like row names (string/factor)
      if (is.character(dt[[first_col_name]]) || is.factor(dt[[first_col_name]])) {
           log_info(sprintf("  Using first column '%s' as row names for %s", first_col_name, basename(f_path)))
           feature_names_loaded <- dt[[first_col_name]] # Use first_col_name

           # Check for duplicates before assigning
           if(any(duplicated(feature_names_loaded))) {
               log_warning("  Duplicate row names found in the first column of ", basename(f_path))
           }

           dt[, (first_col_name) := NULL] # Remove the column used for row names
           mat <- as.matrix(dt)
           rownames(mat) <- feature_names_loaded
           rm(feature_names_loaded)
      } else {
           log_warning(sprintf("  First column of %s does not look like row names. Assuming no row names.", basename(f_path)))
           mat <- as.matrix(dt)
      }
      storage.mode(mat) <- "double"
      result[[title]] <- mat
      log_info(sprintf("  Loaded %s: shape (%s)", basename(f_path), paste(dim(result[[title]]), collapse=", ")))
      rm(dt, mat); gc()
    }, error = function(e) {
      # Use f_path or base_filename in the error message, f_name might not exist
      log_warning(sprintf("  Failed to load or process %s: %s", f_path, e$message))
      result[[title]] <- NULL
    })
  } else {
    # Use f_path in the warning message
    log_warning(sprintf("  File not found: %s", f_path))
    result[[title]] <- NULL
  }

  log_info("Finished loading requested precomputed result(s)."); gc()
  # Return NULL if beta loading failed
  return(if (!is.null(result[[title]])) result else NULL)
}


# calculate_and_report_correlation - UNCHANGED
calculate_and_report_correlation <- function(mat1, mat2, name1 = "Mat1", name2 = "Mat2", threshold = 0.99) {
  passed <- FALSE; correlation <- NA_real_; message_out <- sprintf("Correlation check between %s and %s", name1, name2)
  if (is.null(mat1) || is.null(mat2)) { message_out <- paste(message_out, "- SKIPPED (Missing Data)"); log_warning(message_out); cat_print(message_out); return(list(passed = passed, correlation = correlation, message = message_out)) }
  if (!is.matrix(mat1) && !is.array(mat1)) { log_warning(sprintf("Input '%s' is not a matrix/array. Attempting conversion.", name1)); mat1 <- tryCatch(as.matrix(mat1), error = function(e) NULL); if (is.null(mat1)) { message_out <- paste(message_out, "- FAILED (Cannot convert mat1 to matrix)"); cat_print(message_out); return(list(passed=F, correlation=NA, message=message_out)) } }
  if (!is.matrix(mat2) && !is.array(mat2)) { log_warning(sprintf("Input '%s' is not a matrix/array. Attempting conversion.", name2)); mat2 <- tryCatch(as.matrix(mat2), error = function(e) NULL); if (is.null(mat2)) { message_out <- paste(message_out, "- FAILED (Cannot convert mat2 to matrix)"); cat_print(message_out); return(list(passed=F, correlation=NA, message=message_out)) } }
  shape_mismatch <- FALSE
  if (!all(dim(mat1) == dim(mat2))) { log_warning(sprintf("  Shape mismatch: %s (%s), %s (%s).", name1, paste(dim(mat1), collapse=", "), name2, paste(dim(mat2), collapse=", "))); shape_mismatch <- TRUE
    if(length(mat1) != length(mat2)) { message_out <- paste(message_out, "- FAILED (Element Count Mismatch)"); log_error(message_out) }
    log_warning("  Attempting correlation on flattened arrays despite shape mismatch.")
  }
  flat1 <- as.vector(mat1); flat2 <- as.vector(mat2)
  valid_mask <- is.finite(flat1) & is.finite(flat2); num_valid <- sum(valid_mask)
  if (num_valid < 2) {
      message_out <- paste(message_out, sprintf(": R=NaN (Insufficient valid data points: %d) - FAILED", num_valid)); log_warning(message_out); cat_print(message_out)
      rm(flat1, flat2, valid_mask); return(list(passed = passed, correlation = correlation, message = message_out))
  }
  flat1_valid <- flat1[valid_mask]; flat2_valid <- flat2[valid_mask]
  rm(flat1, flat2, valid_mask)
  sd1 <- sd(flat1_valid); sd2 <- sd(flat2_valid)
  if (is.na(sd1) || is.na(sd2)) { message_out <- paste(message_out, ": R=NaN (NA in SD calculation) - FAILED"); log_warning(message_out); cat_print(message_out); rm(flat1_valid, flat2_valid); return(list(passed = passed, correlation = correlation, message = message_out)) }
  if (sd1 < 1e-10 || sd2 < 1e-10) { is_c1 <- sd1 < 1e-10; is_c2 <- sd2 < 1e-10
    if (is_c1 && is_c2) { if (all(abs(flat1_valid - flat2_valid) < 1e-9)) { correlation <- 1.0; passed <- TRUE; message_out <- paste(message_out, ": R=1.0 (Identical Const) - PASSED"); log_info(message_out) } else { correlation <- NaN; message_out <- paste(message_out, "- FAILED (Different Const)"); log_warning(message_out) }
    } else { correlation <- NaN; message_out <- paste(message_out, "- FAILED (One Const, One Non-Const)"); log_warning(message_out) }
    rm(flat1_valid, flat2_valid); cat_print(message_out); return(list(passed = passed, correlation = correlation, message = message_out))
  }
  cor_result <- tryCatch( cor(flat1_valid, flat2_valid, method = "pearson"), error = function(e) { log_error(sprintf(" Correlation Error (%s vs %s): %s", name1, name2, e$message)); return(NA_real_) } )
  rm(flat1_valid, flat2_valid)
  if (is.na(cor_result)) { message_out <- paste(message_out, ": R=NaN - FAILED (Invalid Corr Result)"); log_warning(message_out)
  } else { correlation <- cor_result; message_out <- paste(message_out, sprintf(": R=%+.6f", correlation))
    if (correlation >= threshold) { passed <- TRUE; message_out <- paste(message_out, sprintf(" (>= %.4f) - PASSED", threshold)); log_info(message_out)
    } else { message_out <- paste(message_out, sprintf(" (< %.4f) - FAILED THRESHOLD", threshold)); log_warning(message_out) }
  }
  cat_print(message_out); if (shape_mismatch) message_out <- paste(message_out, "[Shape Mismatch Warning]"); return(list(passed = passed, correlation = correlation, message = message_out))
}


# --- Estimate optimal batch size based on GPU memory and matrix dimensions ---
estimate_optimal_batch_size <- function(n_genes, n_features, n_samples, n_rand, is_sparse, nnz = NULL, target_usage = 0.7) {
  # Get available GPU memory
  mem_info <- get_cuda_memory_info()
  free_mem <- mem_info$free_memory
  total_mem <- mem_info$total_memory
  
  # Calculate target memory (70% of available by default)
  target_mem <- free_mem * target_usage
  
  log_info(sprintf("GPU memory: %.2f GB total, %.2f GB free, targeting %.2f GB usage", 
                 total_mem / (1024^3), free_mem / (1024^3), target_mem / (1024^3)))
  
  # Try different batch sizes to find optimal
  batch_sizes <- c(n_samples, n_samples/2, n_samples/4, n_samples/8, 50, 25, 10, 5)
  batch_sizes <- unique(round(batch_sizes))
  batch_sizes <- batch_sizes[batch_sizes > 0 & batch_sizes <= n_samples]
  
  for(bs in batch_sizes) {
    mem_req <- estimate_cuda_memory(
      n_genes = n_genes,
      n_features = n_features, 
      n_samples = n_samples,
      n_rand = n_rand,
      nnz = nnz,
      is_sparse = is_sparse,
      batch_size = bs
    )
    
    if(mem_req$required_bytes < target_mem) {
      log_info(sprintf("Selected batch_size = %d (requires %.2f GB, %.1f%% of available memory)",
                     bs, mem_req$required_bytes / (1024^3), 
                     100 * mem_req$required_bytes / free_mem))
      return(bs)
    }
  }
  
  # If we get here, even the smallest batch size is too large
  log_warning("Even minimal batch size exceeds target memory. Using batch_size = 1")
  return(1)
}


# --- Main Script Logic ---

# Initialize CUDA
log_info("Initializing CUDA...")
init_result <- RidgeRegCuda::check_cuda_available(CUDA_DEVICE_ID)
if (!init_result$status == 0) { # Check status code
    log_error("CUDA initialization failed: ", init_result$message)
}
log_info("CUDA initialized successfully.")

# Configure memory options
if(ENABLE_MEMORY_POOLING) {
  log_info("Configuring CUDA memory pooling...")
  tryCatch({
    prev_state <- set_cuda_memory_options(
      enable_pool = TRUE,
      allocation_size = MEMORY_POOL_SIZE_MB,
      release_threshold = MEMORY_RELEASE_THRESHOLD_MB
    )
    log_info("Memory pooling enabled successfully.")
  }, error = function(e) {
    log_warning("Failed to configure memory pooling: ", e$message)
  })
}

# --- Load Data ---
log_info("Loading data...")
suppressPackageStartupMessages(library(data.table)) # Ensure data.table is loaded

# --- Load Expression Data (Y) using data.table::fread ---
y_text_path <- file.path(data_dir, "GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz") # <-- Path to text file
log_info(sprintf("Loading expression data (Y) from text file: %s", y_text_path))

if (!file.exists(y_text_path)) {
    log_error("Expression text file not found: ", y_text_path)
}

Y_df <- NULL # Initialize Y_df
tryCatch({
    # Using fread for potentially faster reading of text files
    Y_df_dt <- data.table::fread(y_text_path, sep = '\t', header = TRUE, check.names = FALSE)

    # Assuming first column contains gene names for Y
    y_genes_col_name <- names(Y_df_dt)[1]
    log_info(sprintf("Assuming first column ('%s') of expression file contains gene names.", y_genes_col_name))
    y_genes <- Y_df_dt[[y_genes_col_name]]

    if(any(duplicated(y_genes))) {
        log_warning("Duplicate gene names found in Y's first column. Rownames may not be unique.")
    }

    # Create data frame excluding the first column, then assign rownames
    Y_df <- as.data.frame(Y_df_dt[, -1])
    rownames(Y_df) <- y_genes

    log_info(sprintf("Loaded Y: shape (%s)", paste(dim(Y_df), collapse=", ")))
    rm(Y_df_dt, y_genes); gc() # Clean up data.table and gene vector

}, error = function(e) {
    log_error("Failed to read or process expression text file: ", y_text_path, " - Error: ", e$message)
})

log_info(sprintf("Loading signature matrix (X) from: %s", sig_matrix_path))
X_df_dt <- data.table::fread(sig_matrix_path)
x_genes <- X_df_dt$V1
X_df <- as.data.frame(X_df_dt[, -1])
rownames(X_df) <- x_genes
log_info(sprintf("Loaded X: shape (%s)", paste(dim(X_df), collapse=", ")))
rm(X_df_dt, x_genes); gc()


# --- Preprocessing: Filter to Common Genes --- # 
log_info("Preprocessing: Filtering Y and X to common genes...")

if (is.null(Y_df) || is.null(X_df)) {
    log_error("Y_df or X_df not loaded correctly before filtering.")
}
if (is.null(rownames(Y_df)) || is.null(rownames(X_df))) {
    log_error("Rownames missing from Y_df or X_df before filtering. Cannot find common genes.")
}

common_genes <- intersect(rownames(Y_df), rownames(X_df))
n_common <- length(common_genes)

if (n_common == 0) {
    log_error("No common genes found between Y and X after loading.")
}
log_info(sprintf("Found %d common genes. Filtering matrices.", n_common))

# Create the filtered data frames using Y_df and X_df
Y_df_filtered <- Y_df[common_genes, , drop = FALSE]
X_df_filtered <- X_df[common_genes, , drop = FALSE]

# --- Add checks after filtering ---
if(!is.data.frame(Y_df_filtered) || nrow(Y_df_filtered) != n_common){
     log_error("Failed to create Y_df_filtered correctly.")
}
if(!is.data.frame(X_df_filtered) || nrow(X_df_filtered) != n_common){
     log_error("Failed to create X_df_filtered correctly.")
}
# --- End checks ---


# Remove original large dataframes immediately after filtering
rm(Y_df, X_df); gc()
log_info(sprintf("Filtered data shapes: Y=(%s), X=(%s)",
                 paste(dim(Y_df_filtered), collapse=", "), paste(dim(X_df_filtered), collapse=", ")))

# --- Preprocessing: Convert Filtered DataFrames to Matrices --- # <<< THIS IS WHERE THE ERROR OCCURRED
log_info("Preprocessing: Converting filtered dataframes to matrices...")

# --- Preprocessing: Convert to Matrix (No R Scaling Yet) ---
log_info("Preprocessing: Converting dataframes to matrices...")

# Convert Y dataframe to matrix
Y_mat_unscaled <- as.matrix(Y_df_filtered) # Convert the filtered dataframe
storage.mode(Y_mat_unscaled) <- "double"   # Ensure numeric type is double
y_sample_names <- colnames(Y_mat_unscaled) # Store column names (samples)
y_gene_names <- rownames(Y_mat_unscaled)   # Store row names (genes)
rm(Y_df_filtered); gc()                    # Remove dataframe to save memory

# Convert X dataframe to matrix
X_mat_unscaled <- as.matrix(X_df_filtered) # Convert the filtered dataframe
storage.mode(X_mat_unscaled) <- "double"   # Ensure numeric type is double
x_feature_names <- colnames(X_mat_unscaled) # Store column names (features)
x_gene_names <- rownames(X_mat_unscaled)   # Store row names (genes)
rm(X_df_filtered); gc()                    # Remove dataframe to save memory

# --- Optional: Add validation checks here ---
if(!identical(y_gene_names, x_gene_names)) {
    log_warning("Gene names (rownames) between Y and X matrices do not match after conversion/filtering! Check filtering logic.")
    # Consider stopping if they MUST match: log_error("Rownames mismatch...")
}
if(anyNA(Y_mat_unscaled) || any(!is.finite(Y_mat_unscaled))) {
    log_warning("Y matrix contains NA or non-finite values after conversion.")
    # Consider imputation or error handling
}
if(anyNA(X_mat_unscaled) || any(!is.finite(X_mat_unscaled))) {
    log_warning("X matrix contains NA or non-finite values after conversion.")
    # Consider imputation or error handling
}
# --- End validation ---

# --- Preprocessing: Scale using CUDA ---
log_info("Preprocessing: Performing COLUMN-WISE scaling using CUDA...")

# Scale X (Dense)
log_info("  Scaling X (dense) using CUDA...")
scale_result_X <- NULL
tryCatch({
    scale_result_X <- RidgeRegCuda::scale_dense_matrix_cuda(X_mat_unscaled, CUDA_DEVICE_ID)
}, error = function(e) {
    log_error("CUDA scaling of dense matrix X failed: ", e$message)
})
if (is.null(scale_result_X)) log_error("CUDA scaling of X returned NULL.")
X_final <- scale_result_X$scaled_matrix
center_X <- scale_result_X$center # Store if needed
scale_X <- scale_result_X$scale   # Store if needed
log_info(sprintf("Scaled X shape: (%s)", paste(dim(X_final), collapse=", ")))
# --- Verify dimnames after CUDA scaling ---
if(!identical(rownames(X_final), x_gene_names) || !identical(colnames(X_final), x_feature_names)){
    log_warning("Dimnames mismatch after dense CUDA scaling for X!")
    # Attempt to reapply?
    # dimnames(X_final) <- list(x_gene_names, x_feature_names)
}
rm(X_mat_unscaled, scale_result_X); gc()


# Scale Y (Sparse-Aware - Requires converting to dgCMatrix first)
log_info("  Scaling Y (sparse-aware) using CUDA...")
log_info("    Converting Y to dgCMatrix...")
Y_sparse_unscaled <- as(Y_mat_unscaled, "dgCMatrix")
rm(Y_mat_unscaled); gc()

scale_result_Y <- NULL
tryCatch({
    scale_result_Y <- RidgeRegCuda::scale_sparse_matrix_csc_cuda(Y_sparse_unscaled, CUDA_DEVICE_ID)
}, error = function(e) {
    log_error("CUDA scaling of sparse matrix Y failed: ", e$message)
})
if (is.null(scale_result_Y)) log_error("CUDA scaling of Y returned NULL.")

# Note: scale_sparse_matrix_csc_cuda now returns the reconstructed matrix
Y_final_sparse_scaled <- scale_result_Y$scaled_matrix # This is dgCMatrix
center_Y_nz <- scale_result_Y$`scaled:center-nz` # Store if needed
scale_Y_nz <- scale_result_Y$`scaled:scale-nz`   # Store if needed
log_info(sprintf("Scaled Y (sparse) shape: (%s)", paste(dim(Y_final_sparse_scaled), collapse=", ")))
# --- Verify dimnames after CUDA scaling ---
if(!identical(rownames(Y_final_sparse_scaled), y_gene_names) || !identical(colnames(Y_final_sparse_scaled), y_sample_names)){
    log_warning("Dimnames mismatch after sparse CUDA scaling for Y!")
    # The reconstruction inside the R function should handle this. Double check that logic.
}
rm(Y_sparse_unscaled, scale_result_Y); gc()

log_info("Using scaled sparse Y (dgCMatrix) as input for ridge_cuda.")

# --- IMPORTANT: Decide which Y to use for ridge_cuda ---
# If ridge_cuda_sparse is used, pass the CSC components from Y_final_sparse_scaled
# If ridge_cuda_dense is used, convert Y_final_sparse_scaled back to dense matrix
log_info("Converting scaled sparse Y back to dense matrix for ridge_cuda_dense...")
Y_final <- as.matrix(Y_final_sparse_scaled)
rm(Y_final_sparse_scaled) # Remove sparse version if using dense ridge
storage.mode(Y_final) <- "double"
gc()
log_info(sprintf("Final dense Y shape after sparse scaling & conversion: (%s)", paste(dim(Y_final), collapse=", ")))

# --- Final Checks Before Ridge ---
log_info("Storing final feature and sample names...")
feature_names <- colnames(X_final) # Should be from CUDA scaling
sample_names <- colnames(Y_final) # Should be from CUDA scaling
if (is.null(feature_names) || length(feature_names) != ncol(X_final)) log_error("Final feature_names NULL or length mismatch after CUDA scale!")
if (is.null(sample_names) || length(sample_names) != ncol(Y_final)) log_error("Final sample_names NULL or length mismatch after CUDA scale!")
if (is.null(rownames(X_final)) || is.null(rownames(Y_final)) || !identical(rownames(X_final), rownames(Y_final))) log_error("Final rownames mismatch between X and Y after CUDA scale!")

log_info(sprintf("Final matrix shapes before ridge_cuda: Y=(%s), X=(%s)", paste(dim(Y_final), collapse=", "), paste(dim(X_final), collapse=", ")))
if (any(!is.finite(Y_final))) log_error("Y_final contains invalid values after CUDA scaling.")
if (any(!is.finite(X_final))) log_error("X_final contains invalid values after CUDA scaling.")
gc()

# --- Determine optimal batch size ---
n_genes <- nrow(X_final)
n_features <- ncol(X_final)
n_samples <- ncol(Y_final)

batch_size <- NULL
if (AUTO_BATCH_SIZE) {
    log_info("Determining optimal batch size based on GPU memory...")
    is_Y_sparse <- inherits(Y_final, "dgCMatrix")
    nnz_value <- if(is_Y_sparse) Matrix::nnzero(Y_final) else NULL
    
    batch_size <- estimate_optimal_batch_size(
        n_genes = n_genes,
        n_features = n_features,
        n_samples = n_samples,
        n_rand = N_RAND_PERM,
        is_sparse = FALSE, # Y_final is dense at this point
        nnz = NULL,
        target_usage = MEMORY_USAGE_TARGET
    )
} else {
    batch_size <- min(FIXED_BATCH_SIZE, n_samples)
    log_info(sprintf("Using configured fixed batch size: %d", batch_size))
}

# --- Run Ridge Regression with Batch Processing ---
log_info(sprintf("\n--- Running Permutation Test with Batch Processing (RidgeRegCuda pkg, nrand=%d, lambda=%.1f, batch_size=%d) ---", 
                N_RAND_PERM, LAMBDA_VAL, batch_size))
results_perm_pkg <- NULL
start_time_pkg_perm <- Sys.time()
computed_beta_matrix <- NULL # Initialize variable

tryCatch({
  if (!exists("X_final") || !exists("Y_final")) {
      stop("Input matrices X_final or Y_final not found before ridge_cuda call.")
  }

  # Use batch processing in ridge_cuda call
  results_perm_pkg <- RidgeRegCuda::ridge_cuda(
    X = X_final, 
    Y = Y_final, 
    lambda = LAMBDA_VAL,
    n_rand = N_RAND_PERM, 
    batch_size = batch_size,  # Add batch_size parameter
    device_id = CUDA_DEVICE_ID
  )

  duration_pkg_perm <- difftime(Sys.time(), start_time_pkg_perm, units = "secs")
  log_info(sprintf("Package permutation test finished in %.3f seconds.", duration_pkg_perm))

  # --- Create named beta matrix ---
  if (!is.null(results_perm_pkg) && results_perm_pkg$status == 0) {
      log_info("Permutation test ridge_cuda call successful (status 0).")
      beta_data_from_pkg <- results_perm_pkg$beta
      if (is.null(beta_data_from_pkg)) {
          log_error("Beta matrix is NULL in results_perm_pkg.")
          results_valid <- FALSE
      } else {
          log_info("Creating new beta matrix with names from scaled inputs...")
          # Use feature_names and sample_names stored AFTER CUDA scaling
          if (is.null(feature_names) || length(feature_names) != nrow(beta_data_from_pkg)) {
              log_error(sprintf("Mismatch or NULL feature_names: Expected %d, Got %d.", nrow(beta_data_from_pkg), length(feature_names)))
              results_valid <- FALSE
          } else if (is.null(sample_names) || length(sample_names) != ncol(beta_data_from_pkg)) {
              log_error(sprintf("Mismatch or NULL sample_names: Expected %d, Got %d.", ncol(beta_data_from_pkg), length(sample_names)))
              results_valid <- FALSE
          } else {
              computed_beta_matrix <- matrix(
                  as.numeric(beta_data_from_pkg),
                  nrow = length(feature_names),
                  ncol = length(sample_names),
                  byrow = FALSE,
                  dimnames = list(feature_names, sample_names) # Use names from scaled X/Y
              )
              storage.mode(computed_beta_matrix) <- "double"
              log_info("First 6 rownames of computed beta matrix (using names from scaled inputs):")
              print(head(rownames(computed_beta_matrix), 6))
              results_valid <- TRUE
          }
      }
      rm(beta_data_from_pkg); gc()
      # --- END BETA CREATION ---


      # --- SAVE COMPUTED BETA (using the newly created computed_beta_matrix) ---
      if (results_valid && !is.null(computed_beta_matrix)) {
          log_info(sprintf("Saving computed beta matrix (shape: %s) to: %s",
                           paste(dim(computed_beta_matrix), collapse=","), output_file))
          beta_dt <- data.table::as.data.table(computed_beta_matrix, keep.rownames = "Feature")
          tryCatch({
              data.table::fwrite(beta_dt, file = output_file, sep = ",", row.names = FALSE, col.names = TRUE, compress="none") # Save uncompressed
              log_info("Computed beta matrix saved successfully.")
          }, error = function(e) {
              log_error("Failed to save computed beta matrix to ", output_file, ": ", e$message)
              results_perm_pkg <- NULL
              computed_beta_matrix <- NULL
          })
          rm(beta_dt); gc()
      } else {
          log_error("Cannot save beta matrix because results are invalid or matrix creation failed.")
          results_perm_pkg <- NULL
          computed_beta_matrix <- NULL
      }
      # --- END SAVE ---

  } else if (!is.null(results_perm_pkg)) {
       log_warning("Permutation test ridge_cuda call returned non-zero status: ", results_perm_pkg$status, " - ", attr(results_perm_pkg, "message"))
       results_perm_pkg <- NULL
  } else {
      log_warning("Permutation test ridge_cuda call returned NULL.")
      results_perm_pkg <- NULL
  }

}, error = function(e) {
  log_error("Permutation test ridge_cuda call failed: ", e$message)
  results_perm_pkg <- NULL
})

# --- CRITICAL MEMORY MANAGEMENT STEP ---
log_info("Removing large input matrices X_final and Y_final from memory...")
if (exists("Y_final")) { rm(Y_final); log_info("Removed Y_final.") } else { log_warning("Y_final not found for removal.")}
if (exists("X_final")) { rm(X_final); log_info("Removed X_final.") } else { log_warning("X_final not found for removal.")}
gc()
log_info("Input matrices removed. Current memory usage:")
print(gc())

# --- Load Precomputed Results ---
precomputed_beta_list <- load_beta_csv_result(precomputed_dir, precomputed_filename)


# --- Compare Results (Only Beta) ---
cat_print("\nComparing Beta Matrix results (package) with precomputed data:")
beta_check_passed <- FALSE
comparison_performed <- FALSE

if (!is.null(precomputed_beta_list) && !is.null(precomputed_beta_list$beta) &&
    !is.null(computed_beta_matrix)) {

    comparison_performed <- TRUE
    log_info("Performing comparison for Beta matrix using newly created computed beta...")

    pkg_beta_full <- computed_beta_matrix
    precomp_beta <- precomputed_beta_list$beta

    log_info(sprintf("Initial dimensions: Package Beta=(%s), Precomputed Beta=(%s)",
                     paste(dim(pkg_beta_full), collapse=","), paste(dim(precomp_beta), collapse=",")))

    if ("background" %in% rownames(pkg_beta_full)) {
        log_info("Removing 'background' row from package beta matrix for comparison.")
        pkg_beta <- pkg_beta_full[rownames(pkg_beta_full) != "background", , drop = FALSE]
        log_info(sprintf("Package Beta dimensions after removing background: (%s)",
                         paste(dim(pkg_beta), collapse=",")))
    } else {
        log_warning("'background' row not found in package beta matrix. Comparing as is.")
        pkg_beta <- pkg_beta_full
    }

    # --- Alignment Logic ---
    current_key_mismatch <- FALSE
    common_rows <- NULL; common_cols <- NULL
    pkg_rows <- rownames(pkg_beta); pre_rows <- rownames(precomp_beta)
    pkg_cols <- colnames(pkg_beta); pre_cols <- colnames(precomp_beta)

    # Align Rows
    if (!is.null(pkg_rows) && !is.null(pre_rows)) {
         log_info("First 6 rownames from pkg_beta (background removed):")
         print(head(pkg_rows))
         log_info("First 6 rownames from precomp_beta:")
         print(head(pre_rows))

         common_rows <- intersect(pkg_rows, pre_rows)
         log_info(sprintf("Found %d common rows (features) for Beta comparison.", length(common_rows)))
         if(length(common_rows) < max(length(pkg_rows), length(pre_rows))) {
            log_warning("Row names mismatch/subset for Beta. Using ", length(common_rows), " common rows.")
         }
         if(length(common_rows) == 0) {
             log_error("No common rows found after removing background.")
             current_key_mismatch <- TRUE;
         } else {
            pkg_beta <- pkg_beta[common_rows, , drop = FALSE]
            precomp_beta <- precomp_beta[common_rows, , drop = FALSE]
            pkg_cols <- colnames(pkg_beta); pre_cols <- colnames(precomp_beta)
         }
    } else {
        log_warning("Row names missing from one or both Beta matrices. Cannot align rows by name.")
        if (!all(dim(pkg_beta) == dim(precomp_beta))) {
             current_key_mismatch <- TRUE;
        }
    }
    if (exists("pkg_beta_full")) rm(pkg_beta_full)

    # Align Columns
    if (!current_key_mismatch) {
        if (!is.null(pkg_cols) && !is.null(pre_cols)) {
            common_cols <- intersect(pkg_cols, pre_cols)
            log_info(sprintf("Found %d common columns (samples) for Beta comparison.", length(common_cols)))
            if(length(common_cols) < max(length(pkg_cols), length(pre_cols))) {
                log_warning("Col names mismatch/subset for Beta. Using ", length(common_cols), " common columns.")
            }
            if(length(common_cols) == 0) {
                log_error("No common columns (samples) found.")
                current_key_mismatch <- TRUE;
            } else {
                pkg_beta <- pkg_beta[, common_cols, drop = FALSE]
                precomp_beta <- precomp_beta[, common_cols, drop = FALSE]
            }
        } else {
             log_warning("Col names missing from one or both Beta matrices. Cannot align columns by name.")
             if (!all(dim(pkg_beta) == dim(precomp_beta))) {
                  current_key_mismatch <- TRUE;
             }
        }
    }

    # Final dimension check
    if (!current_key_mismatch && !all(dim(pkg_beta) == dim(precomp_beta))) {
        log_error(sprintf("Shape mismatch for Beta AFTER alignment: Pkg=(%s), Pre=(%s)",
                            paste(dim(pkg_beta), collapse=","), paste(dim(precomp_beta), collapse=",")))
        current_key_mismatch <- TRUE;
    }

    # Perform correlation
    if (!current_key_mismatch) {
         log_info(sprintf("Comparing aligned Beta matrices with dimensions: (%s)", paste(dim(pkg_beta), collapse=",")))
         corr_result <- calculate_and_report_correlation(pkg_beta, precomp_beta, "Pkg Perm Beta", "Precomp Perm Beta", BETA_CORR_THRESHOLD)
         beta_check_passed <- corr_result$passed
    } else {
         log_warning("Skipping Beta correlation due to alignment/shape issues.")
         beta_check_passed <- FALSE
    }
    if(exists("pkg_beta")) rm(pkg_beta)
    if(exists("precomp_beta")) rm(precomp_beta)
    gc()

} else {
    if(is.null(precomputed_beta_list) || is.null(precomputed_beta_list$beta)) log_warning("Skipping Beta comparison - precomputed Beta data missing or failed to load.")
    if(is.null(computed_beta_matrix)) log_warning("Skipping Beta comparison - computed beta matrix was not successfully generated or assigned names.")
    beta_check_passed <- FALSE
}


# --- Clean up remaining large data objects BEFORE final summary ---
log_info("Cleaning up remaining large data objects before exiting...")
if (exists("results_perm_pkg")) rm(results_perm_pkg)
if (exists("computed_beta_matrix")) rm(computed_beta_matrix)
if (exists("precomputed_beta_list")) rm(precomputed_beta_list)
if (exists("feature_names")) rm(feature_names)
if (exists("sample_names")) rm(sample_names)
gc()

# --- Test B: T-test (SKIPPED) ---
log_info("\n--- T-test (nrand=0) SKIPPED ---")

# --- Final Summary ---
cat_print("\n--- Final Summary of Beta Correlation Check ---")
if (!comparison_performed) {
    if (exists("duration_pkg_perm") && !is.null(results_perm_pkg) && results_perm_pkg$status == 0) { # Check if run succeeded but comparison failed
        log_warning("\nPackage RidgeRegCuda run completed and saved beta, but comparison was skipped (check logs).")
        quit(save = "no", status = 1) # Treat comparison failure as script failure
    } else if (exists("duration_pkg_perm")) { # Check if run failed before comparison
         log_warning("\nPackage RidgeRegCuda run completed but failed (status != 0) or saving failed. Comparison skipped.")
         quit(save = "no", status = 1)
    } else { # Run didn't even complete
        log_error("\nPackage RidgeRegCuda run did not complete successfully or produce valid results.")
        quit(save = "no", status = 1)
    }
} else if (beta_check_passed) {
  log_info("Beta matrix correlation check passed minimum threshold.")
  log_info("Note: Sparse Y scaling was used with batch processing.")
  log_info("\nPackage RidgeRegCuda implementation example finished successfully.")
  quit(save = "no", status = 0)
} else {
  log_warning("\nBeta matrix correlation check FAILED or was skipped due to mismatches (check logs above).")
  log_warning("Note: Sparse Y scaling was used with batch processing.")
  log_warning("\nPackage RidgeRegCuda implementation example finished with Beta correlation check failure or mismatch.")
  quit(save = "no", status = 1)
}

# Clean up CUDA resources
tryCatch({
  RidgeRegCuda::cleanup_cuda()
  log_info("CUDA resources cleaned up.")
}, error = function(e) {
  log_warning("Failed to clean up CUDA resources: ", e$message)
})