# File: run_r_secact_analysis.R

# --- INSTALL/LOAD PACKAGES ---
if (!requireNamespace("SecAct", quietly = TRUE)) {
  install.packages("SecAct")
}
if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table")
}
if (!requireNamespace("peakRAM", quietly = TRUE)) {
  install.packages("peakRAM")  # Install the peakRAM package for memory monitoring
}
library(SecAct)
library(data.table)
library(peakRAM)

# --------- UTILITY FUNCTIONS -----------

# Function to read an expression matrix from a file.
# Tries both tab and comma delimiters and assumes the first column contains gene IDs.
read_expression_matrix <- function(file) {
  message("Loading dataset: ", file)
  for (sep in c("\t", ",")) {
    tryCatch({
      expr <- fread(file, sep = sep)
      # Assume first column holds gene IDs; remaining columns are numeric data.
      mat <- as.matrix(expr[, -1, with = FALSE])
      rownames(mat) <- expr[[1]]
      return(mat)
    }, error = function(e) {
      message("  Failed with sep='", sep, "': ", e$message)
    })
  }
  stop("Failed to load expression matrix from file: ", file)
}

# Function to save SecAct inference results and metadata.
save_secact_results <- function(result, expr_data, input_file, output_dir,
                                n_rand, lambda_val, exec_time, peak_mem_mb) {
  dataset_name <- sub("\\.[^.]+$", "", basename(input_file))  # Remove file extension
  
  # Save each result matrix as a CSV.
  write.csv(result$beta,   file.path(output_dir, paste0(dataset_name, "_R_beta.csv")))
  write.csv(result$se,     file.path(output_dir, paste0(dataset_name, "_R_se.csv")))
  write.csv(result$zscore, file.path(output_dir, paste0(dataset_name, "_R_zscore.csv")))
  write.csv(result$pvalue, file.path(output_dir, paste0(dataset_name, "_R_pvalue.csv")))
  
  # Build metadata including execution time and peak memory usage.
  metadata <- data.frame(
    dataset = dataset_name,
    execution_time_sec = exec_time,
    peak_memory_mb = peak_mem_mb,
    n_rand = n_rand,
    lambda = lambda_val,
    n_genes = nrow(expr_data),
    n_samples = ncol(expr_data)
  )
  write.csv(metadata, file.path(output_dir, paste0(dataset_name, "_R_metadata.csv")), row.names = FALSE)
}

# Function to run SecAct.inference under memory monitoring.
# We capture the result in a global temporary variable to avoid peakRAM converting it.
run_secact_analysis <- function(input_file, output_dir, n_rand = 1000,
                                lambda_val = 5e5, sig_matrix = "SecAct") {
  # Ensure the output directory exists.
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Read expression matrix.
  expr_data <- read_expression_matrix(input_file)
  message("Dataset dimensions: ", nrow(expr_data), " genes x ", ncol(expr_data), " samples")
  
  message("Running SecAct.inference with peakRAM monitoring...")
  
  # Use a global temporary variable to capture the result.
  result_global <<- NULL
  metrics <- tryCatch({
    peakRAM({
      result_global <<- SecAct.inference(
        Y = expr_data,
        SigMat = sig_matrix,
        lambda = lambda_val,
        nrand = n_rand
      )
    }, include_result = FALSE)
  }, error = function(e) {
    message("ERROR during SecAct.inference execution: ", e$message)
    return(NULL)
  })
  
  # Retrieve the result from the global variable and then remove it from the global env.
  result <- result_global
  rm(result_global, envir = .GlobalEnv)
  
  if (is.null(result)) {
    stop("SecAct.inference did not produce a valid result.")
  }
  
  exec_time <- metrics$Elapsed_Time_sec[1]
  peak_mem_mb <- metrics$Peak_RAM_Used_MiB[1]
  message("Execution time: ", round(exec_time, 2), " seconds")
  message("Peak memory usage: ", round(peak_mem_mb, 2), " MB")
  
  # Check that result is a list with the expected components.
  if (!is.list(result) || is.null(result$beta) || is.null(result$se) ||
      is.null(result$zscore) || is.null(result$pvalue)) {
    stop("SecAct.inference result does not have the expected components (beta, se, zscore, pvalue).")
  }
  
  # Save the results and metadata.
  save_secact_results(result, expr_data, input_file, output_dir,
                      n_rand, lambda_val, exec_time, peak_mem_mb)
  message("Results saved to: ", output_dir)
  invisible(result)
}

# --------- CONFIG & RUN -----------

datasets_dir <- "/data/parks34/projects/SecActPy/datasets"
results_dir  <- "r_secact_results"

datasets <- list(
  list(name = "GSE100093_IFNG", file = "GSE100093.IFNG.expr.gz", size = "small"),
  list(name = "Pancreatic_Nivolumab", file = "Pancreatic_Nivolumab_Padron2022.logTPM.gz", size = "medium"),
  list(name = "GSE131907_Lung_Cancer", file = "GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz", size = "large")
)

size_to_nrand <- list(small = 1000, medium = 500, large = 100)

for (dataset in datasets) {
  message("\n==== Processing dataset: ", dataset$name, " ====")
  input_file <- file.path(datasets_dir, dataset$file)
  n_rand <- size_to_nrand[[dataset$size]]
  
  tryCatch({
    run_secact_analysis(input_file, results_dir, n_rand = n_rand)
  }, error = function(e) {
    message("Error processing dataset ", dataset$name, ": ", e$message)
  })
}

# --------- SUMMARY -----------
message("\nCreating summary metadata...")
summary_files <- list.files(results_dir, pattern = "_R_metadata.csv", full.names = TRUE)

if (length(summary_files) > 0) {
  # Use fill = TRUE to handle inconsistent columns across metadata files.
  summary_df <- rbindlist(lapply(summary_files, fread), fill = TRUE)
  fwrite(summary_df, file.path(results_dir, "R_summary.csv"))
  message("Summary written to ", file.path(results_dir, "R_summary.csv"))
} else {
  message("No result metadata found to summarize.")
}

message("\nAll SecAct analyses completed successfully.")
