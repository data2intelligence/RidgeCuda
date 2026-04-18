#!/usr/bin/env Rscript

# check_cor.R - Compare computed and precomputed beta matrices

# --- Load Required Libraries ---
suppressPackageStartupMessages({
  library(data.table)
  library(stats)
})

# --- Basic Logging ---
log_info <- function(...) { message(format(Sys.time(), "%Y-%m-%d %H:%M:%S - INFO - "), ...) }
log_warning <- function(...) { warning(format(Sys.time(), "%Y-%m-%d %H:%M:%S - WARNING - "), ..., call. = FALSE) }
log_error <- function(...) { stop(format(Sys.time(), "%Y-%m-%d %H:%M:%S - ERROR - "), ..., call. = FALSE) }
cat_print <- function(...) { cat(..., "\n") }

# --- Configuration ---
# !! ADJUST THESE PATHS !!
computed_beta_file <- "/data/parks34/projects/RidgeInf/src/dense_beta.csv" # File saved by compare_large.R
precomputed_beta_file <- "/data/parks34/projects/RidgeInf/ComparePy2R/r_secact_results/GSE131907_Lung_Cancer_normalized_log2TPM_matrix_R_beta.csv" 

# Correlation threshold
BETA_CORR_THRESHOLD <- 0.999

# --- Helper Function: Load Beta Matrix ---
# Simplified loader focusing only on beta CSV
load_beta_matrix <- function(f_path, matrix_name = "Matrix") {
  log_info(sprintf("Loading %s from: %s", matrix_name, f_path))
  beta_mat <- NULL

  if (!file.exists(f_path)) {
    log_warning(sprintf("  File not found: %s", f_path))
    return(NULL)
  }

  tryCatch({
    dt <- data.table::fread(f_path)
    first_col_name <- names(dt)[1]

    if (is.character(dt[[first_col_name]]) || is.factor(dt[[first_col_name]])) {
         log_info(sprintf("  Using first column '%s' as row names for %s", first_col_name, basename(f_path)))
         feature_names_loaded <- dt[[first_col_name]]

         # --- Handle Duplicates ---
         if(any(duplicated(feature_names_loaded))) {
             log_warning("  Duplicate row names found in ", basename(f_path), ". Making unique using make.unique().")
             feature_names_unique <- make.unique(as.character(feature_names_loaded), sep = ".") # Ensure character and specify separator
         } else {
             feature_names_unique <- feature_names_loaded
         }
         # --- End Handle Duplicates ---

         dt[, (first_col_name) := NULL]
         beta_mat <- as.matrix(dt)
         rownames(beta_mat) <- feature_names_unique # Assign unique names
         rm(feature_names_loaded, feature_names_unique) # Clean up temps

    } else if (is.numeric(dt[[first_col_name]])) {
         log_info(sprintf("  First column '%s' is numeric, using as row names for %s", first_col_name, basename(f_path)))
         feature_names_loaded <- as.character(dt[[first_col_name]])
         
         # --- Handle Duplicates ---
         if(any(duplicated(feature_names_loaded))) {
             log_warning("  Duplicate row names found in ", basename(f_path), ". Making unique using make.unique().")
             feature_names_unique <- make.unique(feature_names_loaded, sep = ".") # Ensure character and specify separator
         } else {
             feature_names_unique <- feature_names_loaded
         }
         # --- End Handle Duplicates ---
         
         dt[, (first_col_name) := NULL]
         beta_mat <- as.matrix(dt)
         rownames(beta_mat) <- feature_names_unique # Assign unique names
         rm(feature_names_loaded, feature_names_unique) # Clean up temps
    } else {
         log_warning(sprintf("  First column of %s does not look like row names. Assuming no row names.", basename(f_path)))
         beta_mat <- as.matrix(dt)
    }
    storage.mode(beta_mat) <- "double"
    log_info(sprintf("  Loaded %s: shape (%s)", basename(f_path), paste(dim(beta_mat), collapse=", ")))
    rm(dt); gc()
  }, error = function(e) {
    log_warning(sprintf("  Failed to load or process %s: %s", f_path, e$message))
    beta_mat <- NULL
  })

  return(beta_mat)
}

# --- Helper Function: Calculate Correlation ---
# (Same as in compare_large.R)
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


# --- Main Logic ---
log_info("--- Starting Beta Matrix Comparison ---")

# Load computed beta
computed_beta <- load_beta_matrix(computed_beta_file, "Computed Beta")

# Load precomputed beta
precomputed_beta <- load_beta_matrix(precomputed_beta_file, "Precomputed Beta")

# Check if loading succeeded
if (is.null(computed_beta)) {
  log_error("Failed to load computed beta matrix from: ", computed_beta_file)
}
if (is.null(precomputed_beta)) {
  log_error("Failed to load precomputed beta matrix from: ", precomputed_beta_file)
}

# --- Alignment Logic (Optional but Recommended) ---
log_info("Aligning matrices by common rows (features) and columns (samples)...")
current_key_mismatch <- FALSE
common_rows <- NULL; common_cols <- NULL

# Get initial dimnames
comp_rows <- rownames(computed_beta); pre_rows <- rownames(precomputed_beta)
comp_cols <- colnames(computed_beta); pre_cols <- colnames(precomputed_beta)

# Align Rows (Features)
if (!is.null(comp_rows) && !is.null(pre_rows)) {
     # Convert row names to character if needed
     comp_rows <- as.character(comp_rows)
     pre_rows <- as.character(pre_rows)
     
     common_rows <- intersect(comp_rows, pre_rows)
     log_info(sprintf("Found %d common rows (features) for comparison.", length(common_rows)))
     if(length(common_rows) < max(length(comp_rows), length(pre_rows))) {
        log_warning("Row names mismatch/subset. Using ", length(common_rows), " common rows.")
     }
     if(length(common_rows) == 0) {
         log_error("No common rows (features) found between matrices.")
         current_key_mismatch <- TRUE;
     } else {
        computed_beta <- computed_beta[common_rows, , drop = FALSE]
        precomputed_beta <- precomputed_beta[common_rows, , drop = FALSE]
        # Update col lists after row subsetting
        comp_cols <- colnames(computed_beta); pre_cols <- colnames(precomputed_beta)
     }
} else {
    log_warning("Row names missing from one or both matrices. Cannot align rows by name. Comparing based on existing dimensions.")
    if (!all(dim(computed_beta) == dim(precomputed_beta))) {
         log_warning("Dimensions mismatch and cannot align by row names.")
         current_key_mismatch <- TRUE;
    }
}

# Align Columns (Samples) - only if rows aligned successfully
if (!current_key_mismatch) {
    if (!is.null(comp_cols) && !is.null(pre_cols)) {
        common_cols <- intersect(comp_cols, pre_cols)
        log_info(sprintf("Found %d common columns (samples) for comparison.", length(common_cols)))
        if(length(common_cols) < max(length(comp_cols), length(pre_cols))) {
            log_warning("Col names mismatch/subset. Using ", length(common_cols), " common columns.")
        }
        if(length(common_cols) == 0) {
            log_error("No common columns (samples) found between matrices.")
            current_key_mismatch <- TRUE;
        } else {
            computed_beta <- computed_beta[, common_cols, drop = FALSE]
            precomputed_beta <- precomputed_beta[, common_cols, drop = FALSE]
        }
    } else {
         log_warning("Col names missing from one or both matrices. Cannot align columns by name. Comparing based on existing dimensions.")
         if (!all(dim(computed_beta) == dim(precomputed_beta))) {
             log_warning("Dimensions mismatch and cannot align by column names.")
             current_key_mismatch <- TRUE;
         }
    }
}

# Final dimension check after alignment
if (!current_key_mismatch && !all(dim(computed_beta) == dim(precomputed_beta))) {
    log_error(sprintf("Shape mismatch AFTER alignment: Computed=(%s), Precomputed=(%s)",
                        paste(dim(computed_beta), collapse=","), paste(dim(precomputed_beta), collapse=",")))
    current_key_mismatch <- TRUE;
}

# --- Calculate and Report Correlation ---
cat_print("\n--- Correlation Results ---")
final_status <- 1 # Default to failure

if (!current_key_mismatch) {
     log_info(sprintf("Comparing aligned matrices with final dimensions: (%s)", paste(dim(computed_beta), collapse=",")))
     corr_result <- calculate_and_report_correlation(
         computed_beta,
         precomputed_beta,
         "Computed Beta",
         "Precomputed Beta",
         BETA_CORR_THRESHOLD
     )
     if(corr_result$passed) {
         final_status <- 0 # Success
         log_info("Beta matrix correlation check PASSED.")
     } else {
         log_warning("Beta matrix correlation check FAILED.")
         final_status <- 1
     }
} else {
     log_warning("Skipping correlation calculation due to alignment/shape issues.")
     final_status <- 1 # Treat mismatch as failure
}

log_info("--- Comparison Finished ---")
quit(save = "no", status = final_status) # Exit with 0 on success, 1 on failure/mismatch