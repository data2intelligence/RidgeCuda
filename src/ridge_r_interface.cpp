/**
 * ridge_r_interface.cpp - R interface for CUDA-accelerated ridge regression
 * Connects R with the CUDA implementation in ridge_cuda.cu
 */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h> // For DllInfo and R_registerRoutines
#include <cstring>         // For strcmp, etc. (though not strictly needed here)
#include <cstdlib>         // For srand / rand (host-side Fisher-Yates)
#include <vector>          // For std::vector in get_cuda_devices_r
#include <cuda_runtime.h>

#include "ridge_cuda.h"    // Include the C/CUDA interface header

// Constants
#define MAX_GPU_NAME_LEN 256 // Consistent naming

// Helper function prototype
static SEXP create_result_list(SEXP beta_r, SEXP se_r, SEXP zscore_r, SEXP pvalue_r,
                              double df_val, int status_code, const char* status_msg);

static const char* ridge_status_msg(int code) {
    switch (code) {
        case 0:   return "Success";
        case -10: return "CUDA not initialized";
        case -11: return "Invalid input dimensions";
        case -12: return "Invalid lambda value";
        case -13: return "NULL pointer provided to C function";
        case -14: return "Invalid n_rand value (must be > 0)";
        case -100: return "Insufficient GPU memory for workload (memory-pressure guard fail-fast)";
        case 1:   return "General CUDA Runtime Error";
        case 2:   return "cuBLAS Error";
        case 3:   return "cuSOLVER Error";
        case 4:   return "cuSPARSE Error";
        case 5:   return "cuRAND Error";
        case 10:  return "Cholesky factorization failed";
        case 11:  return "Linear system solve failed";
        case 12:  return "Host memory allocation failed (permutation)";
        case 13:  return "Host memory allocation failed (permutation stats)";
        default:  return "Unknown error in ridge_cuda backend";
    }
}

//----------------------------------------------------------------------------//
// Environment Management R Interface Functions                             //
//----------------------------------------------------------------------------//

/**
 * @brief R interface to check CUDA availability and initialize.
 *
 * Calls ridge_cuda_init and returns status information.
 *
 * @param device_id_r Integer SEXP containing the desired CUDA device ID.
 * @return A list containing 'status' (integer) and 'message' (character).
 */
SEXP check_cuda_available_r(SEXP device_id_r) {
    // Validate input type
    if (!isInteger(device_id_r) || length(device_id_r) != 1) {
        error("device_id must be a single integer value");
    }
    int device_id = asInteger(device_id_r);

    // Call the C initialization function
    int status = ridge_cuda_init(device_id);

    // --- Prepare the result list ---
    SEXP result_list = PROTECT(allocVector(VECSXP, 2)); // List with 2 elements
    SEXP status_sexp = PROTECT(ScalarInteger(status));

    // Map status code to a user-friendly message
    const char* message;
    switch (status) {
        case 0:  message = "CUDA initialization successful"; break;
        case -1: message = "Failed to get CUDA device count"; break;
        case -2: message = "No CUDA devices found"; break;
        case -3: message = "Invalid device ID requested"; break;
        case -4: message = "Failed to set CUDA device"; break;
        case -5: message = "Failed to create cuBLAS handle"; break;
        case -6: message = "Failed to create cuSOLVER handle"; break;
        case -7: message = "Failed to create cuSPARSE handle"; break;
        case -8: message = "Failed to create cuRAND generator"; break;
        case -9: message = "Failed to set cuRAND seed"; break;
        default: message = "Unknown CUDA initialization error"; // Catch unexpected codes
    }
    SEXP message_sexp = PROTECT(mkString(message));

    // Assign elements to the list
    SET_VECTOR_ELT(result_list, 0, status_sexp);
    SET_VECTOR_ELT(result_list, 1, message_sexp);

    // Set names for the list elements
    SEXP names_sexp = PROTECT(allocVector(STRSXP, 2));
    SET_STRING_ELT(names_sexp, 0, mkChar("status"));
    SET_STRING_ELT(names_sexp, 1, mkChar("message"));
    setAttrib(result_list, R_NamesSymbol, names_sexp);

    UNPROTECT(4); // Unprotect: result_list, status_sexp, message_sexp, names_sexp
    return result_list;
}

/**
 * @brief R interface to get information about available CUDA devices.
 *
 * Calls ridge_cuda_get_devices and formats the output as an R data frame.
 *
 * @param max_devices_r Integer SEXP setting a limit on devices queried (unused by C function).
 * @return A data frame with columns: device_id, name, memory_mb.
 */
SEXP get_cuda_devices_r(SEXP max_devices_r) {
     // Although max_devices_r isn't used by the C function, keep it for interface consistency
    if (!isInteger(max_devices_r) || length(max_devices_r) != 1) {
         error("max_devices must be a single integer value");
    }
    // int max_devices = asInteger(max_devices_r); // Not needed by C function

    int device_count = 0;
    // Query count first to allocate appropriately
    int status = ridge_cuda_get_devices(&device_count, NULL, 0, NULL);
     if (status != 0 && device_count == 0) { // Handle case where count fails AND returns 0 devices
         error("Failed to query CUDA device count (status %d). Is CUDA driver installed?", status);
         return R_NilValue; // Return NULL for error
     }
     if (device_count == 0) { // Handle case where count succeeds but finds 0 devices
         // Return an empty data frame structure
         SEXP result = PROTECT(allocVector(VECSXP, 3));
         SEXP ids = PROTECT(allocVector(INTSXP, 0));
         SEXP names = PROTECT(allocVector(STRSXP, 0));
         SEXP memories = PROTECT(allocVector(REALSXP, 0));
         SET_VECTOR_ELT(result, 0, ids);
         SET_VECTOR_ELT(result, 1, names);
         SET_VECTOR_ELT(result, 2, memories);
         // Set column names, class, row names (empty)
         SEXP colnames = PROTECT(allocVector(STRSXP, 3));
         SET_STRING_ELT(colnames, 0, mkChar("device_id")); SET_STRING_ELT(colnames, 1, mkChar("name")); SET_STRING_ELT(colnames, 2, mkChar("memory_mb"));
         setAttrib(result, R_NamesSymbol, colnames);
         SEXP class_name = PROTECT(mkString("data.frame"));
         setAttrib(result, R_ClassSymbol, class_name);
         setAttrib(result, R_RowNamesSymbol, allocVector(INTSXP, 0));
         UNPROTECT(6);
         return result;
     }


    // Allocate memory based on the actual device_count
    std::vector<char*> device_names_vec(device_count);
    std::vector<size_t> device_memories_vec(device_count);
    std::vector<char> name_buffer( (size_t)device_count * MAX_GPU_NAME_LEN); // Contiguous buffer

    // Assign pointers into the contiguous buffer
    for (int i = 0; i < device_count; ++i) {
        device_names_vec[i] = &name_buffer[ (size_t)i * MAX_GPU_NAME_LEN ];
    }

    // Get the actual device information
    status = ridge_cuda_get_devices(&device_count, device_names_vec.data(), MAX_GPU_NAME_LEN, device_memories_vec.data());

    if (status != 0) {
        // No need to free individual names as they point into name_buffer
        error("Failed to get CUDA device properties (status %d)", status);
        return R_NilValue;
    }

    // --- Create the result data frame ---
    SEXP result_df = PROTECT(allocVector(VECSXP, 3)); // 3 columns
    SEXP ids_sexp = PROTECT(allocVector(INTSXP, device_count));
    SEXP names_sexp = PROTECT(allocVector(STRSXP, device_count));
    SEXP memories_sexp = PROTECT(allocVector(REALSXP, device_count));

    // Populate the R vectors
    for (int i = 0; i < device_count; i++) {
        INTEGER(ids_sexp)[i] = i;
        SET_STRING_ELT(names_sexp, i, mkChar(device_names_vec[i])); // Use names from vector
        REAL(memories_sexp)[i] = (double)device_memories_vec[i] / (1024.0 * 1024.0); // Convert bytes to MB
    }

    // Assign vectors to the list (data frame)
    SET_VECTOR_ELT(result_df, 0, ids_sexp);
    SET_VECTOR_ELT(result_df, 1, names_sexp);
    SET_VECTOR_ELT(result_df, 2, memories_sexp);

    // Set column names
    SEXP colnames_sexp = PROTECT(allocVector(STRSXP, 3));
    SET_STRING_ELT(colnames_sexp, 0, mkChar("device_id"));
    SET_STRING_ELT(colnames_sexp, 1, mkChar("name"));
    SET_STRING_ELT(colnames_sexp, 2, mkChar("memory_mb"));
    setAttrib(result_df, R_NamesSymbol, colnames_sexp);

    // Set class attribute to "data.frame"
    SEXP class_name_sexp = PROTECT(mkString("data.frame")); // Use mkString for single string
    setAttrib(result_df, R_ClassSymbol, class_name_sexp);

    // Set row names (standard 1-based integer sequence)
    SEXP rownames_sexp = PROTECT(allocVector(INTSXP, device_count));
    for (int i = 0; i < device_count; i++) {
        INTEGER(rownames_sexp)[i] = i + 1;
    }
    setAttrib(result_df, R_RowNamesSymbol, rownames_sexp);

    UNPROTECT(7); // result_df, ids_sexp, names_sexp, memories_sexp, colnames_sexp, class_name_sexp, rownames_sexp
    return result_df;
}


/**
 * @brief R interface to get CUDA memory info for the current device.
 *
 * Calls ridge_cuda_get_memory_info and returns free/total memory.
 * Assumes ridge_cuda_init has been called.
 *
 * @param device_id_r Integer SEXP (Note: ignored by C function, uses current device).
 * @return List containing free_memory and total_memory (in bytes).
 */
SEXP ridge_cuda_get_memory_info_r(SEXP device_id_r) {
    // Note: The C function ridge_cuda_get_memory_info doesn't take device_id
    // It operates on the currently initialized device. We keep the R arg for consistency.
    if (!isInteger(device_id_r) || length(device_id_r) != 1) {
        error("device_id must be a single integer value");
    }
    // int device_id = asInteger(device_id_r); // Not used by C function

    size_t free_memory = 0;
    size_t total_memory = 0;
    int status = ridge_cuda_get_memory_info(&free_memory, &total_memory);

    if (status != 0) {
        error("Failed to get CUDA memory information (status %d). Is CUDA initialized?", status);
    }

    // Create result list
    SEXP result = PROTECT(allocVector(VECSXP, 2)); // Changed from 3, removed device_id
    SEXP free_mem_sexp = PROTECT(ScalarReal((double)free_memory));
    SEXP total_mem_sexp = PROTECT(ScalarReal((double)total_memory));

    SET_VECTOR_ELT(result, 0, free_mem_sexp);
    SET_VECTOR_ELT(result, 1, total_mem_sexp);

    // Set names
    SEXP names = PROTECT(allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, mkChar("free_memory"));
    SET_STRING_ELT(names, 1, mkChar("total_memory"));
    setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(4); // result, free_mem_sexp, total_mem_sexp, names
    return result;
}

//----------------------------------------------------------------------------//
// Ridge Regression R Interface Functions                                   //
//----------------------------------------------------------------------------//

/**
 * @brief R interface for dense X, dense Y ridge regression.
 *
 * Extracts data from R objects, calls ridge_cuda_dense, and formats results.
 *
 * @param X_r Numeric matrix SEXP for X.
 * @param Y_r Numeric matrix SEXP for Y.
 * @param lambda_r Numeric SEXP for lambda.
 * @param n_rand_r Integer SEXP for number of permutations (0 for t-test).
 * @param batch_size_r Integer SEXP for batch size (number of Y columns per batch).
 * @param device_id_r Integer SEXP for CUDA device ID.
 * @return List containing results (beta, se, zscore, pvalue, df, status, message).
 */
SEXP ridge_cuda_dense_r(SEXP X_r, SEXP Y_r, SEXP lambda_r, SEXP n_rand_r, SEXP batch_size_r, SEXP device_id_r) {
    // --- Input Validation ---
    if (!isMatrix(X_r) || !isReal(X_r)) error("X must be a numeric matrix");
    if (!isMatrix(Y_r) || !isReal(Y_r)) error("Y must be a numeric matrix");
    if (!isReal(lambda_r) || length(lambda_r) != 1) error("lambda must be a single numeric value");
    if (!isInteger(n_rand_r) || length(n_rand_r) != 1) error("n_rand must be a single integer value");
    if (!isInteger(batch_size_r) || length(batch_size_r) != 1) error("batch_size must be a single integer value");
    if (!isInteger(device_id_r) || length(device_id_r) != 1) error("device_id must be a single integer value");

    // --- Extract Dimensions ---
    SEXP dim_X = getAttrib(X_r, R_DimSymbol);
    SEXP dim_Y = getAttrib(Y_r, R_DimSymbol);
    int n_genes = INTEGER(dim_X)[0];
    int n_features = INTEGER(dim_X)[1];
    int n_samples = INTEGER(dim_Y)[1];
    if (INTEGER(dim_Y)[0] != n_genes) { error("X and Y must have the same number of rows (n_genes)"); }

    // --- Extract Parameters ---
    double lambda_val = REAL(lambda_r)[0];
    int n_rand = INTEGER(n_rand_r)[0];
    int batch_size = INTEGER(batch_size_r)[0];
    int device_id = asInteger(device_id_r);
    if (lambda_val < 0.0) error("lambda must be non-negative");
    // --- T-TEST NO LONGER SUPPORTED, n_rand=0 is invalid for dense now ---
    if (n_rand <= 0) error("n_rand must be positive for permutation testing (t-test removed)");
    
    // --- Validate batch_size ---
    if (batch_size < 0) {
        warning("Negative batch_size provided, using automatic batch sizing");
        batch_size = 0; // Let CUDA code handle automatic sizing
    }

    // --- Initialize CUDA ---
    int init_status = ridge_cuda_init(device_id);
    if (init_status != 0) { error("Failed to initialize CUDA (error %d)", init_status); }

    // --- Get Data Pointers ---
    // Assuming COLUMN-MAJOR from R, matching .cu file modifications
    double* X_ptr = REAL(X_r);
    double* Y_ptr = REAL(Y_r);

    // --- Allocate Output R Objects ---
    SEXP beta_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP se_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP zscore_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP pvalue_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    double* beta_ptr = REAL(beta_r);
    double* se_ptr = REAL(se_r);
    double* zscore_ptr = REAL(zscore_r);
    double* pvalue_ptr = REAL(pvalue_r);

    // Initialize output arrays with NA
    size_t output_size = (size_t)n_features * n_samples;
    for (size_t i = 0; i < output_size; i++) {
        beta_ptr[i] = NA_REAL; se_ptr[i] = NA_REAL;
        zscore_ptr[i] = NA_REAL; pvalue_ptr[i] = NA_REAL;
    }

    // --- Call the CUDA Implementation ---
    int status = ridge_cuda_dense(X_ptr, Y_ptr, n_genes, n_features, n_samples,
                               lambda_val, n_rand, batch_size,
                               beta_ptr, se_ptr, zscore_ptr, pvalue_ptr,
                               NULL /* no perm_table: use fisher_yates */);

    SEXP result = create_result_list(beta_r, se_r, zscore_r, pvalue_r,
                                     NA_REAL, status, ridge_status_msg(status));
    UNPROTECT(4);
    return result;
}

/**
 * @brief R interface for dense X, sparse Y (dgCMatrix) ridge regression.
 *
 * Extracts data from R objects, calls ridge_cuda_sparse, and formats results.
 * Handles the CSC format of dgCMatrix.
 *
 * @param X_r Numeric matrix SEXP for X.
 * @param Y_r dgCMatrix SEXP for Y (must inherit from "dgCMatrix").
 * @param lambda_r Numeric SEXP for lambda.
 * @param n_rand_r Integer SEXP for number of permutations (0 for t-test).
 * @param batch_size_r Integer SEXP for batch size (number of Y columns per batch).
 * @param device_id_r Integer SEXP for CUDA device ID.
 * @return List containing results (beta, se, zscore, pvalue, df=NA, status, message).
 */
SEXP ridge_cuda_sparse_r(SEXP X_r, SEXP Y_r, SEXP lambda_r, SEXP n_rand_r, SEXP batch_size_r, SEXP device_id_r) {
    // --- Input Validation ---
    if (!isMatrix(X_r) || !isReal(X_r)) error("X must be a numeric matrix");
    if (!inherits(Y_r, "dgCMatrix")) error("Y must be a dgCMatrix object from the Matrix package");
    if (!isReal(lambda_r) || length(lambda_r) != 1) error("lambda must be a single numeric value");
    if (!isInteger(n_rand_r) || length(n_rand_r) != 1) error("n_rand must be a single integer value");
    if (!isInteger(batch_size_r) || length(batch_size_r) != 1) error("batch_size must be a single integer value");
    if (!isInteger(device_id_r) || length(device_id_r) != 1) error("device_id must be a single integer value");

    // --- Extract CSC Components ---
    SEXP y_vals_sexp = PROTECT(R_do_slot(Y_r, install("x")));
    SEXP y_col_ptr_sexp = PROTECT(R_do_slot(Y_r, install("p")));
    SEXP y_row_ind_sexp = PROTECT(R_do_slot(Y_r, install("i")));
    SEXP y_dims_sexp = PROTECT(R_do_slot(Y_r, install("Dim")));
    if (!isReal(y_vals_sexp)) { UNPROTECT(4); error("dgCMatrix 'x' slot is not numeric"); }
    if (!isInteger(y_col_ptr_sexp)) { UNPROTECT(4); error("dgCMatrix 'p' slot is not integer"); }
    if (!isInteger(y_row_ind_sexp)) { UNPROTECT(4); error("dgCMatrix 'i' slot is not integer"); }
    if (!isInteger(y_dims_sexp) || length(y_dims_sexp) != 2) { UNPROTECT(4); error("dgCMatrix 'Dim' slot is invalid"); }

    // --- Extract Dimensions ---
    SEXP dim_X = getAttrib(X_r, R_DimSymbol);
    int n_genes = INTEGER(dim_X)[0];
    int n_features = INTEGER(dim_X)[1];
    int n_samples = INTEGER(y_dims_sexp)[1];
    int nnz = length(y_vals_sexp);
    if (INTEGER(y_dims_sexp)[0] != n_genes) { UNPROTECT(4); error("X and sparse Y must have the same number of rows (n_genes)"); }

    // --- Extract Parameters ---
    double lambda_val = REAL(lambda_r)[0];
    int n_rand = INTEGER(n_rand_r)[0];
    int batch_size = INTEGER(batch_size_r)[0];
    int device_id = asInteger(device_id_r);
    if (lambda_val < 0.0) { UNPROTECT(4); error("lambda must be non-negative"); }
    // --- T-TEST NO LONGER SUPPORTED, n_rand=0 is invalid for sparse now ---
    if (n_rand <= 0) { UNPROTECT(4); error("n_rand must be positive for permutation testing (t-test removed)"); }
    
    // --- Validate batch_size ---
    if (batch_size < 0) {
        warning("Negative batch_size provided, using automatic batch sizing");
        batch_size = 0; // Let CUDA code handle automatic sizing
    }

    // --- Initialize CUDA ---
    int init_status = ridge_cuda_init(device_id);
    if (init_status != 0) { UNPROTECT(4); error("Failed to initialize CUDA (error %d)", init_status); }

    // --- Get Data Pointers ---
    // Assuming COLUMN-MAJOR from R, matching .cu file modifications
    double* X_ptr = REAL(X_r);
    double* Y_vals_ptr = REAL(y_vals_sexp);
    int* Y_col_ptr_r = INTEGER(y_col_ptr_sexp);
    int* Y_row_ind_r = INTEGER(y_row_ind_sexp);

    // --- Allocate Output R Objects ---
    SEXP beta_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP se_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP zscore_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP pvalue_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    double* beta_ptr = REAL(beta_r);
    double* se_ptr = REAL(se_r);
    double* zscore_ptr = REAL(zscore_r);
    double* pvalue_ptr = REAL(pvalue_r);
    size_t output_size = (size_t)n_features * n_samples;
    for (size_t i = 0; i < output_size; i++) {
        beta_ptr[i] = NA_REAL; se_ptr[i] = NA_REAL;
        zscore_ptr[i] = NA_REAL; pvalue_ptr[i] = NA_REAL;
    }

    // --- Call the CUDA Implementation ---
    // No perm_table or col-normalization here — that's the perm-aware
    // variant (ridge_cuda_sparse_with_perm_r, below).
    int status = ridge_cuda_sparse(X_ptr, n_genes, n_features,
                                Y_vals_ptr, Y_row_ind_r, Y_col_ptr_r,
                                n_samples, nnz, lambda_val, n_rand, batch_size,
                                beta_ptr, se_ptr, zscore_ptr, pvalue_ptr,
                                NULL /* perm_table */,
                                NULL /* col_mu */, NULL /* col_sigma */);

    SEXP result = create_result_list(beta_r, se_r, zscore_r, pvalue_r,
                                     NA_REAL, status, ridge_status_msg(status));
    UNPROTECT(8);
    return result;
}

/**
 * @brief R interface to clean up CUDA resources.
 *
 * Calls ridge_cuda_cleanup.
 *
 * @return Logical SEXP (TRUE).
 */
SEXP cleanup_cuda_r(void) {
    ridge_cuda_cleanup();
    return ScalarLogical(1); // Return TRUE (success is assumed unless error above)
}

//----------------------------------------------------------------------------//
// Advanced/Utility R Interface Functions                                   //
//----------------------------------------------------------------------------//

/**
 * @brief R interface to estimate GPU memory requirements.
 */
SEXP ridge_cuda_memory_requirements_r(SEXP n_genes_r, SEXP n_features_r, SEXP n_samples_r,
                                    SEXP nnz_r, SEXP is_sparse_r, SEXP n_rand_r, SEXP batch_size_r) {
    // --- Input Validation & Extraction ---
    if (!isInteger(n_genes_r) || length(n_genes_r)!=1) error("n_genes must be single integer");
    if (!isInteger(n_features_r) || length(n_features_r)!=1) error("n_features must be single integer");
    if (!isInteger(n_samples_r) || length(n_samples_r)!=1) error("n_samples must be single integer");
    if (!isInteger(nnz_r) || length(nnz_r)!=1) error("nnz must be single integer");
    if (!isLogical(is_sparse_r) || length(is_sparse_r)!=1) error("is_sparse must be single logical");
    if (!isInteger(n_rand_r) || length(n_rand_r)!=1) error("n_rand must be single integer");
    if (!isInteger(batch_size_r) || length(batch_size_r)!=1) error("batch_size must be single integer");

    int n_genes = asInteger(n_genes_r);
    int n_features = asInteger(n_features_r);
    int n_samples = asInteger(n_samples_r);
    int nnz = asInteger(nnz_r);
    int is_sparse = asLogical(is_sparse_r); // Convert logical to int (0 or 1)
    int n_rand = asInteger(n_rand_r);
    int batch_size = asInteger(batch_size_r);

    // --- Declare variable BEFORE use ---
    size_t required_bytes = 0; // <<< DECLARE HERE

    // --- Calculate required bytes ---
    if (n_genes <= 0 || n_features <= 0 || n_samples <= 0 || nnz < 0 || n_rand < 0) {
        warning("Invalid non-positive input dimension/count provided to memory requirements check. Returning 0 bytes required.");
        // required_bytes is already 0 from declaration
    } else {
       // Call C function to get the requirement with batch_size
        required_bytes = ridge_cuda_memory_requirements(n_genes, n_features, n_samples, nnz, is_sparse, n_rand, batch_size);
    }

    // --- Create Result List ---
    SEXP result = PROTECT(allocVector(VECSXP, 7)); // Now 7 elements including batch_size
    // Reuse input parameters for the list structure
    SET_VECTOR_ELT(result, 0, n_genes_r);
    SET_VECTOR_ELT(result, 1, n_features_r);
    SET_VECTOR_ELT(result, 2, n_samples_r);
    SET_VECTOR_ELT(result, 3, n_rand_r);
    SET_VECTOR_ELT(result, 4, is_sparse_r);
    SET_VECTOR_ELT(result, 5, batch_size_r); // Add batch_size to result
    // Note: PROTECT the ScalarReal directly when assigning
    SET_VECTOR_ELT(result, 6, PROTECT(ScalarReal((double)required_bytes)));

    // Set names
    SEXP names = PROTECT(allocVector(STRSXP, 7)); // Now 7 names
    SET_STRING_ELT(names, 0, mkChar("n_genes"));
    SET_STRING_ELT(names, 1, mkChar("n_features"));
    SET_STRING_ELT(names, 2, mkChar("n_samples"));
    SET_STRING_ELT(names, 3, mkChar("n_rand"));
    SET_STRING_ELT(names, 4, mkChar("is_sparse"));
    SET_STRING_ELT(names, 5, mkChar("batch_size")); // Add batch_size name
    SET_STRING_ELT(names, 6, mkChar("required_bytes"));
    setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(3); // result, ScalarReal for bytes, names
    return result;
}


/**
 * @brief R interface to set CUDA memory management options.
 *
 * Configures CUDA memory pooling behavior, which can improve performance
 * for repeated allocations by reducing memory fragmentation.
 *
 * @param enable_pool_r Logical SEXP indicating whether to enable memory pooling.
 * @param allocation_size_r Numeric SEXP specifying initial pool size in MB (0 = default).
 * @param release_threshold_r Numeric SEXP specifying release threshold in MB (0 = default).
 * @return Logical SEXP indicating previous pooling state.
 */
SEXP ridge_cuda_set_memory_options_r(SEXP enable_pool_r, SEXP allocation_size_r, SEXP release_threshold_r) {
    // Validate input types
    if (!isLogical(enable_pool_r) || length(enable_pool_r) != 1) {
        error("enable_pool must be a single logical value");
    }
    if (!isReal(allocation_size_r) || length(allocation_size_r) != 1) {
        error("allocation_size must be a single numeric value (in MB)");
    }
    if (!isReal(release_threshold_r) || length(release_threshold_r) != 1) {
        error("release_threshold must be a single numeric value (in MB)");
    }
    
    // Extract values
    int enable_pool = asLogical(enable_pool_r);
    double allocation_size_mb = asReal(allocation_size_r);
    double release_threshold_mb = asReal(release_threshold_r);
    
    // Convert MB to bytes
    size_t allocation_size = allocation_size_mb > 0 ? (size_t)(allocation_size_mb * 1024 * 1024) : 0;
    size_t release_threshold = release_threshold_mb > 0 ? (size_t)(release_threshold_mb * 1024 * 1024) : 0;
    
    // Call C function
    int previous_state = ridge_cuda_set_memory_options(
        enable_pool,
        allocation_size,
        release_threshold
    );
    
    // Handle error codes
    if (previous_state < 0) {
        switch (previous_state) {
            case -1:
                error("CUDA not initialized. Call check_cuda_available() first.");
                break;
            case -2:
                error("Failed to get current CUDA device.");
                break;
            case -3:
                error("Failed to create CUDA memory pool.");
                break;
            case -4:
                error("Failed to set default memory pool for device.");
                break;
            case -5:
                error("Failed to set CUDA memory limits.");
                break;
            default:
                error("Unknown error occurred when setting memory options.");
        }
    }
    
    // Return previous state as logical
    return ScalarLogical(previous_state);
}

/**
 * @brief R interface to set CUDA asynchronous execution mode.
 *
 * Enables or disables asynchronous execution of CUDA operations,
 * which can improve performance by overlapping computation and data transfer.
 *
 * @param enable_async_r Logical SEXP indicating whether to enable async mode.
 * @return Logical SEXP indicating previous async state.
 */
SEXP ridge_cuda_set_async_mode_r(SEXP enable_async_r) {
    // Validate input type
    if (!isLogical(enable_async_r) || length(enable_async_r) != 1) {
        error("enable_async must be a single logical value");
    }
    
    // Extract value
    int enable_async = asLogical(enable_async_r);
    
    // Call C function
    int previous_mode = ridge_cuda_set_async_mode(enable_async);
    
    // Handle error code
    if (previous_mode < 0) {
        error("CUDA not initialized. Call check_cuda_available() first.");
    }
    
    // Return previous mode as logical
    return ScalarLogical(previous_mode);
}

// --- Helper to copy R numeric vector to device ---
static int copy_vector_to_device(SEXP vec_r, double** d_vec, size_t* len = nullptr) {
    if (!isReal(vec_r)) return -1; // Type error
    size_t n = length(vec_r);
    if (len) *len = n;
    double* h_vec = REAL(vec_r);
    cudaError_t cudaStat = cudaMalloc((void**)d_vec, n * sizeof(double));
    if (cudaStat != cudaSuccess) return 1; // CUDA alloc error
    cudaStat = cudaMemcpy(*d_vec, h_vec, n * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess) { cudaFree(*d_vec); *d_vec = nullptr; return 2; } // CUDA copy error
    return 0; // Success
}

// --- R interface for Dense Scaling ---
SEXP ridge_cuda_scale_dense_matrix_r(SEXP matrix_r, SEXP device_id_r) {
    // --- Input Validation ---
    if (!isMatrix(matrix_r) || !isReal(matrix_r)) error("matrix_r must be a numeric matrix");
    if (!isInteger(device_id_r) || length(device_id_r) != 1) error("device_id must be a single integer value");

    // --- Initialize CUDA ---
    int device_id = asInteger(device_id_r);
    int init_status = ridge_cuda_init(device_id);
    if (init_status != 0) { error("Failed to initialize CUDA (error %d)", init_status); }

    // --- Extract Info ---
    SEXP dim_r = getAttrib(matrix_r, R_DimSymbol);
    int n_rows = INTEGER(dim_r)[0];
    int n_cols = INTEGER(dim_r)[1];
    size_t n_elements = (size_t)n_rows * n_cols;
    double* h_matrix_ptr = REAL(matrix_r); // R matrices are column-major

    // --- Allocate GPU Memory ---
    double *d_matrix = NULL, *d_means = NULL, *d_sds = NULL;
    SEXP result_list = R_NilValue; // Initialize result
    int protect_count = 0;

    cudaError_t cudaStat;
    cudaStat = cudaMalloc((void**)&d_matrix, n_elements * sizeof(double));
    if (cudaStat != cudaSuccess) { error("CUDA Error: Failed to allocate memory for d_matrix (%s)", cudaGetErrorString(cudaStat)); }
    cudaStat = cudaMalloc((void**)&d_means, n_cols * sizeof(double));
    if (cudaStat != cudaSuccess) { cudaFree(d_matrix); error("CUDA Error: Failed to allocate memory for d_means (%s)", cudaGetErrorString(cudaStat)); }
    cudaStat = cudaMalloc((void**)&d_sds, n_cols * sizeof(double));
    if (cudaStat != cudaSuccess) { cudaFree(d_matrix); cudaFree(d_means); error("CUDA Error: Failed to allocate memory for d_sds (%s)", cudaGetErrorString(cudaStat)); }

    // --- Copy Host to Device ---
    cudaStat = cudaMemcpy(d_matrix, h_matrix_ptr, n_elements * sizeof(double), cudaMemcpyHostToDevice);
     if (cudaStat != cudaSuccess) {
        cudaFree(d_matrix); cudaFree(d_means); cudaFree(d_sds);
        error("CUDA Error: Failed to copy matrix to device (%s)", cudaGetErrorString(cudaStat));
     }

    // --- Call C/CUDA Scaling Function ---
    int scale_status = ridge_cuda_scale_dense_matrix(d_matrix, n_rows, n_cols, d_means, d_sds);

    if (scale_status == 0) {
        // --- Copy Results Back to Host ---
        SEXP scaled_matrix_r = PROTECT(allocMatrix(REALSXP, n_rows, n_cols)); protect_count++;
        SEXP means_r = PROTECT(allocVector(REALSXP, n_cols)); protect_count++;
        SEXP sds_r = PROTECT(allocVector(REALSXP, n_cols)); protect_count++;

        cudaStat = cudaMemcpy(REAL(scaled_matrix_r), d_matrix, n_elements * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStat != cudaSuccess) warning("CUDA memcpy error getting scaled matrix: %s", cudaGetErrorString(cudaStat));
        cudaStat = cudaMemcpy(REAL(means_r), d_means, n_cols * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStat != cudaSuccess) warning("CUDA memcpy error getting means: %s", cudaGetErrorString(cudaStat));
        cudaStat = cudaMemcpy(REAL(sds_r), d_sds, n_cols * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStat != cudaSuccess) warning("CUDA memcpy error getting sds: %s", cudaGetErrorString(cudaStat));

        // --- Create Result List ---
        result_list = PROTECT(allocVector(VECSXP, 3)); protect_count++;
        SET_VECTOR_ELT(result_list, 0, scaled_matrix_r);
        SET_VECTOR_ELT(result_list, 1, means_r);
        SET_VECTOR_ELT(result_list, 2, sds_r);

        SEXP names_r = PROTECT(allocVector(STRSXP, 3)); protect_count++;
        SET_STRING_ELT(names_r, 0, mkChar("scaled_matrix"));
        SET_STRING_ELT(names_r, 1, mkChar("center")); // Match scale() output names
        SET_STRING_ELT(names_r, 2, mkChar("scale"));
        setAttrib(result_list, R_NamesSymbol, names_r);

        // --- Preserve Dimnames ---
        SEXP original_dimnames = getAttrib(matrix_r, R_DimNamesSymbol);
        if (original_dimnames != R_NilValue) {
            setAttrib(scaled_matrix_r, R_DimNamesSymbol, duplicate(original_dimnames));
        }
        SEXP original_colnames = R_NilValue;
        if (inherits(original_dimnames, "list") && length(original_dimnames) >= 2) {
             original_colnames = VECTOR_ELT(original_dimnames, 1);
             if (original_colnames != R_NilValue && length(original_colnames) == n_cols) {
                setAttrib(means_r, R_NamesSymbol, duplicate(original_colnames));
                setAttrib(sds_r, R_NamesSymbol, duplicate(original_colnames));
             }
        }


    } else {
        error("ridge_cuda_scale_dense_matrix failed with status %d", scale_status);
        // result_list remains R_NilValue
    }

    // --- Cleanup GPU Memory ---
    cudaFree(d_matrix);
    cudaFree(d_means);
    cudaFree(d_sds);

    if (protect_count > 0) UNPROTECT(protect_count);
    return result_list;
}


// --- R interface for Sparse Scaling (CSC) ---
SEXP ridge_cuda_scale_sparse_matrix_csc_r(SEXP Y_r, SEXP device_id_r) {
    // --- Input Validation ---
    if (!inherits(Y_r, "dgCMatrix")) error("Y_r must be a dgCMatrix object");
    if (!isInteger(device_id_r) || length(device_id_r) != 1) error("device_id must be a single integer");

    // --- Initialize CUDA ---
    int device_id = asInteger(device_id_r);
    int init_status = ridge_cuda_init(device_id);
    if (init_status != 0) { error("Failed to initialize CUDA (error %d)", init_status); }

    // --- Extract CSC Components & Info ---
    SEXP y_vals_sexp = PROTECT(R_do_slot(Y_r, install("x")));
    SEXP y_col_ptr_sexp = PROTECT(R_do_slot(Y_r, install("p")));
    SEXP y_row_ind_sexp = PROTECT(R_do_slot(Y_r, install("i")));
    SEXP y_dims_sexp = PROTECT(R_do_slot(Y_r, install("Dim")));
    int protect_count = 4; // Initial protected items

    if (!isReal(y_vals_sexp)) { UNPROTECT(protect_count); error("dgCMatrix 'x' slot is not numeric"); }
    if (!isInteger(y_col_ptr_sexp)) { UNPROTECT(protect_count); error("dgCMatrix 'p' slot is not integer"); }
    if (!isInteger(y_row_ind_sexp)) { UNPROTECT(protect_count); error("dgCMatrix 'i' slot is not integer"); }
    if (!isInteger(y_dims_sexp) || length(y_dims_sexp) != 2) { UNPROTECT(protect_count); error("dgCMatrix 'Dim' slot is invalid"); }

    int n_rows = INTEGER(y_dims_sexp)[0];
    int n_cols = INTEGER(y_dims_sexp)[1];
    int nnz = length(y_vals_sexp);

    double* h_vals_ptr = REAL(y_vals_sexp);
    int* h_col_ptr = INTEGER(y_col_ptr_sexp);
    int* h_row_ind_ptr = INTEGER(y_row_ind_sexp);

    // --- Allocate GPU Memory ---
    double *d_vals = NULL, *d_means_nz = NULL, *d_sds_nz = NULL;
    int *d_row_indices = NULL, *d_col_pointers = NULL; // Keep const correctness in mind later
    SEXP result_list = R_NilValue;

    cudaError_t cudaStat;
    // Alloc data
    cudaStat = cudaMalloc((void**)&d_vals, nnz * sizeof(double));
    if (cudaStat != cudaSuccess) { UNPROTECT(protect_count); error("CUDA Error: Failed to allocate d_vals (%s)", cudaGetErrorString(cudaStat)); }
    cudaStat = cudaMalloc((void**)&d_row_indices, nnz * sizeof(int));
    if (cudaStat != cudaSuccess) { cudaFree(d_vals); UNPROTECT(protect_count); error("CUDA Error: Failed to allocate d_row_indices (%s)", cudaGetErrorString(cudaStat)); }
    cudaStat = cudaMalloc((void**)&d_col_pointers, (n_cols + 1) * sizeof(int));
    if (cudaStat != cudaSuccess) { cudaFree(d_vals); cudaFree(d_row_indices); UNPROTECT(protect_count); error("CUDA Error: Failed to allocate d_col_pointers (%s)", cudaGetErrorString(cudaStat)); }
    // Alloc outputs
    cudaStat = cudaMalloc((void**)&d_means_nz, n_cols * sizeof(double));
    if (cudaStat != cudaSuccess) { /* free others */ cudaFree(d_vals); cudaFree(d_row_indices); cudaFree(d_col_pointers); UNPROTECT(protect_count); error("CUDA Error: Failed to allocate d_means_nz (%s)", cudaGetErrorString(cudaStat)); }
    cudaStat = cudaMalloc((void**)&d_sds_nz, n_cols * sizeof(double));
    if (cudaStat != cudaSuccess) { /* free others */ cudaFree(d_vals); cudaFree(d_row_indices); cudaFree(d_col_pointers); cudaFree(d_means_nz); UNPROTECT(protect_count); error("CUDA Error: Failed to allocate d_sds_nz (%s)", cudaGetErrorString(cudaStat)); }

    // --- Copy Host to Device ---
    cudaStat = cudaMemcpy(d_vals, h_vals_ptr, nnz * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess) { /* free all */ cudaFree(d_vals); cudaFree(d_row_indices); cudaFree(d_col_pointers); cudaFree(d_means_nz); cudaFree(d_sds_nz); UNPROTECT(protect_count); error("CUDA Error: Failed H->D copy d_vals (%s)", cudaGetErrorString(cudaStat));}
    cudaStat = cudaMemcpy(d_row_indices, h_row_ind_ptr, nnz * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess) { /* free all */ cudaFree(d_vals); cudaFree(d_row_indices); cudaFree(d_col_pointers); cudaFree(d_means_nz); cudaFree(d_sds_nz); UNPROTECT(protect_count); error("CUDA Error: Failed H->D copy d_row_indices (%s)", cudaGetErrorString(cudaStat));}
    cudaStat = cudaMemcpy(d_col_pointers, h_col_ptr, (n_cols + 1) * sizeof(int), cudaMemcpyHostToDevice);
     if (cudaStat != cudaSuccess) { /* free all */ cudaFree(d_vals); cudaFree(d_row_indices); cudaFree(d_col_pointers); cudaFree(d_means_nz); cudaFree(d_sds_nz); UNPROTECT(protect_count); error("CUDA Error: Failed H->D copy d_col_pointers (%s)", cudaGetErrorString(cudaStat));}

    // --- Call C/CUDA Sparse Scaling Function ---
    int scale_status = ridge_cuda_scale_sparse_matrix_csc(
        d_vals, d_row_indices, d_col_pointers, n_rows, n_cols, nnz,
        d_means_nz, d_sds_nz, NULL // Not requesting counts back here
    );


    if (scale_status == 0) {
        // --- Copy Results Back to Host ---
        // Only need to copy back the modified vals and the means/sds
        SEXP scaled_vals_r = PROTECT(allocVector(REALSXP, nnz)); protect_count++;
        SEXP means_nz_r = PROTECT(allocVector(REALSXP, n_cols)); protect_count++;
        SEXP sds_nz_r = PROTECT(allocVector(REALSXP, n_cols)); protect_count++;

        cudaStat = cudaMemcpy(REAL(scaled_vals_r), d_vals, nnz * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStat != cudaSuccess) warning("CUDA memcpy error getting scaled vals: %s", cudaGetErrorString(cudaStat));
        cudaStat = cudaMemcpy(REAL(means_nz_r), d_means_nz, n_cols * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStat != cudaSuccess) warning("CUDA memcpy error getting means_nz: %s", cudaGetErrorString(cudaStat));
        cudaStat = cudaMemcpy(REAL(sds_nz_r), d_sds_nz, n_cols * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStat != cudaSuccess) warning("CUDA memcpy error getting sds_nz: %s", cudaGetErrorString(cudaStat));


        // --- Create Result List ---
        // We return the components needed to reconstruct the dgCMatrix in R, plus stats
        result_list = PROTECT(allocVector(VECSXP, 6)); protect_count++;
        SET_VECTOR_ELT(result_list, 0, scaled_vals_r);      // Modified 'x' slot
        SET_VECTOR_ELT(result_list, 1, y_row_ind_sexp);     // Original 'i' slot (unmodified)
        SET_VECTOR_ELT(result_list, 2, y_col_ptr_sexp);     // Original 'p' slot (unmodified)
        SET_VECTOR_ELT(result_list, 3, y_dims_sexp);        // Original 'Dim' slot (unmodified)
        SET_VECTOR_ELT(result_list, 4, means_nz_r);         // Non-zero means
        SET_VECTOR_ELT(result_list, 5, sds_nz_r);           // Non-zero sds

        SEXP names_r = PROTECT(allocVector(STRSXP, 6)); protect_count++;
        SET_STRING_ELT(names_r, 0, mkChar("x"));
        SET_STRING_ELT(names_r, 1, mkChar("i"));
        SET_STRING_ELT(names_r, 2, mkChar("p"));
        SET_STRING_ELT(names_r, 3, mkChar("Dim"));
        SET_STRING_ELT(names_r, 4, mkChar("scaled:center-nz"));
        SET_STRING_ELT(names_r, 5, mkChar("scaled:scale-nz"));
        setAttrib(result_list, R_NamesSymbol, names_r);

        // --- Preserve Colnames for stats ---
        SEXP original_dimnames = R_do_slot(Y_r, install("Dimnames"));
        if (inherits(original_dimnames, "list") && length(original_dimnames) >= 2) {
             SEXP original_colnames = VECTOR_ELT(original_dimnames, 1);
             if (original_colnames != R_NilValue && length(original_colnames) == n_cols) {
                setAttrib(means_nz_r, R_NamesSymbol, duplicate(original_colnames));
                setAttrib(sds_nz_r, R_NamesSymbol, duplicate(original_colnames));
             }
        }

    } else {
         error("ridge_cuda_scale_sparse_matrix_csc failed with status %d", scale_status);
        // result_list remains R_NilValue
    }


    // --- Cleanup GPU Memory ---
    cudaFree(d_vals);
    cudaFree(d_row_indices);
    cudaFree(d_col_pointers);
    cudaFree(d_means_nz);
    cudaFree(d_sds_nz);

    UNPROTECT(protect_count);
    return result_list;
}



//----------------------------------------------------------------------------//
// Helper Functions                                                           //
//----------------------------------------------------------------------------//

/**
 * @brief Creates the standard list object returned by the R interface functions.
 *
 * Encapsulates the results and status information into a named list.
 *
 * @param beta_r SEXP for the beta matrix.
 * @param se_r SEXP for the standard error matrix.
 * @param zscore_r SEXP for the z-score/t-stat matrix.
 * @param pvalue_r SEXP for the p-value matrix.
 * @param df_val Double value for degrees of freedom (can be NA_REAL).
 * @param status_code Integer status code from the C function.
 * @param status_msg Character string describing the status.
 * @return A named list SEXP.
 */
static SEXP create_result_list(SEXP beta_r, SEXP se_r, SEXP zscore_r, SEXP pvalue_r,
                              double df_val, int status_code, const char* status_msg) {

    // Create the list SEXP with 6 elements
    SEXP result_list = PROTECT(allocVector(VECSXP, 6));

    // Create scalar SEXPs for df, status, and message
    SEXP df_r = PROTECT(ScalarReal(df_val));        // Handles NA_REAL correctly
    SEXP status_r = PROTECT(ScalarInteger(status_code));
    SEXP message_r = PROTECT(mkString(status_msg));

    // Assign the SEXPs to the list elements
    SET_VECTOR_ELT(result_list, 0, beta_r);   // Assumes beta_r is already protected
    SET_VECTOR_ELT(result_list, 1, se_r);     // Assumes se_r is already protected
    SET_VECTOR_ELT(result_list, 2, zscore_r); // Assumes zscore_r is already protected
    SET_VECTOR_ELT(result_list, 3, pvalue_r); // Assumes pvalue_r is already protected
    SET_VECTOR_ELT(result_list, 4, df_r);
    SET_VECTOR_ELT(result_list, 5, status_r);

    // Create and set names for the list elements
    SEXP names = PROTECT(allocVector(STRSXP, 6));
    SET_STRING_ELT(names, 0, mkChar("beta"));
    SET_STRING_ELT(names, 1, mkChar("se"));
    SET_STRING_ELT(names, 2, mkChar("zscore"));
    SET_STRING_ELT(names, 3, mkChar("pvalue"));
    SET_STRING_ELT(names, 4, mkChar("df"));
    SET_STRING_ELT(names, 5, mkChar("status"));
    setAttrib(result_list, R_NamesSymbol, names);

    // Add the status message as an attribute (optional but helpful)
    setAttrib(result_list, install("message"), message_r);

    // Unprotect the locally created SEXPs: result_list, df_r, status_r, message_r, names
    UNPROTECT(5);
    return result_list; // Return the protected list
}


/**
 * @brief R interface: dense ridge with caller-supplied permutation table.
 *
 * Variant of ridge_cuda_dense_r that accepts an explicit permutation
 * table (integer matrix, n_rand x n_genes, 0-indexed). Enables bitwise
 * parity with CPU backends when the table is generated by MT19937.
 */
SEXP ridge_cuda_dense_with_perm_r(SEXP X_r, SEXP Y_r, SEXP lambda_r,
                                   SEXP n_rand_r, SEXP batch_size_r,
                                   SEXP device_id_r, SEXP perm_table_r) {
    if (!isMatrix(X_r) || !isReal(X_r)) error("X must be a numeric matrix");
    if (!isMatrix(Y_r) || !isReal(Y_r)) error("Y must be a numeric matrix");
    if (!isReal(lambda_r) || length(lambda_r) != 1) error("lambda must be a single numeric value");
    if (!isInteger(n_rand_r) || length(n_rand_r) != 1) error("n_rand must be a single integer value");
    if (!isInteger(batch_size_r) || length(batch_size_r) != 1) error("batch_size must be a single integer value");
    if (!isInteger(device_id_r) || length(device_id_r) != 1) error("device_id must be a single integer value");
    if (!isMatrix(perm_table_r) || !isInteger(perm_table_r)) error("perm_table must be an integer matrix");

    SEXP dim_X = getAttrib(X_r, R_DimSymbol);
    SEXP dim_Y = getAttrib(Y_r, R_DimSymbol);
    SEXP dim_P = getAttrib(perm_table_r, R_DimSymbol);
    int n_genes = INTEGER(dim_X)[0];
    int n_features = INTEGER(dim_X)[1];
    int n_samples = INTEGER(dim_Y)[1];
    if (INTEGER(dim_Y)[0] != n_genes) error("X and Y must have the same number of rows");

    double lambda_val = REAL(lambda_r)[0];
    int n_rand = INTEGER(n_rand_r)[0];
    int batch_size = INTEGER(batch_size_r)[0];
    int device_id = asInteger(device_id_r);
    if (lambda_val < 0.0) error("lambda must be non-negative");
    if (n_rand <= 0) error("n_rand must be positive");

    if (INTEGER(dim_P)[0] != n_rand) error("perm_table must have n_rand rows");
    if (INTEGER(dim_P)[1] != n_genes) error("perm_table must have n_genes columns");

    if (batch_size < 0) batch_size = 0;

    int init_status = ridge_cuda_init(device_id);
    if (init_status != 0) error("Failed to initialize CUDA (error %d)", init_status);

    double* X_ptr = REAL(X_r);
    double* Y_ptr = REAL(Y_r);

    /* perm_table from R is column-major (n_rand x n_genes). The CUDA
       kernel expects row-major (n_rand rows of n_genes indices each).
       Transpose into a local buffer. */
    int* perm_rowmajor = (int*)malloc((size_t)n_rand * n_genes * sizeof(int));
    if (!perm_rowmajor) error("Failed to allocate perm_table buffer");
    int* perm_cm = INTEGER(perm_table_r);
    for (int r = 0; r < n_rand; r++) {
        for (int j = 0; j < n_genes; j++) {
            perm_rowmajor[(size_t)r * n_genes + j] = perm_cm[(size_t)j * n_rand + r];
        }
    }

    SEXP beta_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP se_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP zscore_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP pvalue_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    double* beta_ptr = REAL(beta_r);
    double* se_ptr = REAL(se_r);
    double* zscore_ptr = REAL(zscore_r);
    double* pvalue_ptr = REAL(pvalue_r);
    size_t output_size = (size_t)n_features * n_samples;
    for (size_t i = 0; i < output_size; i++) {
        beta_ptr[i] = NA_REAL; se_ptr[i] = NA_REAL;
        zscore_ptr[i] = NA_REAL; pvalue_ptr[i] = NA_REAL;
    }

    int status = ridge_cuda_dense(X_ptr, Y_ptr, n_genes, n_features, n_samples,
                                  lambda_val, n_rand, batch_size,
                                  beta_ptr, se_ptr, zscore_ptr, pvalue_ptr,
                                  perm_rowmajor);
    free(perm_rowmajor);

    SEXP result = create_result_list(beta_r, se_r, zscore_r, pvalue_r,
                                     NA_REAL, status, ridge_status_msg(status));
    UNPROTECT(4);
    return result;
}


/**
 * @brief R interface: sparse ridge with caller-supplied permutation table.
 *
 * Variant of ridge_cuda_sparse_r that accepts an explicit permutation
 * table. Routes a dgCMatrix Y through the cuSPARSE-based ridge_cuda_sparse
 * kernel using the caller's perm stream, giving cross-backend bitwise
 * reproducibility for the SPARSE compute path (no host-side densify).
 */
SEXP ridge_cuda_sparse_with_perm_r(SEXP X_r, SEXP Y_r, SEXP lambda_r,
                                    SEXP n_rand_r, SEXP batch_size_r,
                                    SEXP device_id_r, SEXP perm_table_r) {
    // --- Input Validation ---
    if (!isMatrix(X_r) || !isReal(X_r)) error("X must be a numeric matrix");
    if (!inherits(Y_r, "dgCMatrix")) error("Y must be a dgCMatrix object from the Matrix package");
    if (!isReal(lambda_r) || length(lambda_r) != 1) error("lambda must be a single numeric value");
    if (!isInteger(n_rand_r) || length(n_rand_r) != 1) error("n_rand must be a single integer value");
    if (!isInteger(batch_size_r) || length(batch_size_r) != 1) error("batch_size must be a single integer value");
    if (!isInteger(device_id_r) || length(device_id_r) != 1) error("device_id must be a single integer value");
    if (!isMatrix(perm_table_r) || !isInteger(perm_table_r)) error("perm_table must be an integer matrix");

    SEXP y_vals_sexp = PROTECT(R_do_slot(Y_r, install("x")));
    SEXP y_col_ptr_sexp = PROTECT(R_do_slot(Y_r, install("p")));
    SEXP y_row_ind_sexp = PROTECT(R_do_slot(Y_r, install("i")));
    SEXP y_dims_sexp = PROTECT(R_do_slot(Y_r, install("Dim")));
    if (!isReal(y_vals_sexp))    { UNPROTECT(4); error("dgCMatrix 'x' slot is not numeric"); }
    if (!isInteger(y_col_ptr_sexp)) { UNPROTECT(4); error("dgCMatrix 'p' slot is not integer"); }
    if (!isInteger(y_row_ind_sexp)) { UNPROTECT(4); error("dgCMatrix 'i' slot is not integer"); }
    if (!isInteger(y_dims_sexp) || length(y_dims_sexp) != 2) { UNPROTECT(4); error("dgCMatrix 'Dim' slot is invalid"); }

    SEXP dim_X = getAttrib(X_r, R_DimSymbol);
    SEXP dim_P = getAttrib(perm_table_r, R_DimSymbol);
    int n_genes = INTEGER(dim_X)[0];
    int n_features = INTEGER(dim_X)[1];
    int n_samples = INTEGER(y_dims_sexp)[1];
    int nnz = length(y_vals_sexp);
    if (INTEGER(y_dims_sexp)[0] != n_genes) { UNPROTECT(4); error("X and sparse Y must have the same number of rows (n_genes)"); }

    double lambda_val = REAL(lambda_r)[0];
    int n_rand = INTEGER(n_rand_r)[0];
    int batch_size = INTEGER(batch_size_r)[0];
    int device_id = asInteger(device_id_r);
    if (lambda_val < 0.0)   { UNPROTECT(4); error("lambda must be non-negative"); }
    if (n_rand <= 0)        { UNPROTECT(4); error("n_rand must be positive (t-test removed for sparse)"); }
    if (INTEGER(dim_P)[0] != n_rand)   { UNPROTECT(4); error("perm_table must have n_rand rows"); }
    if (INTEGER(dim_P)[1] != n_genes)  { UNPROTECT(4); error("perm_table must have n_genes columns"); }
    if (batch_size < 0) batch_size = 0;

    int init_status = ridge_cuda_init(device_id);
    if (init_status != 0) { UNPROTECT(4); error("Failed to initialize CUDA (error %d)", init_status); }

    double* X_ptr        = REAL(X_r);
    double* Y_vals_ptr   = REAL(y_vals_sexp);
    int*    Y_col_ptr_r  = INTEGER(y_col_ptr_sexp);
    int*    Y_row_ind_r  = INTEGER(y_row_ind_sexp);

    /* perm_table from R is column-major (n_rand x n_genes). The CUDA
       kernel expects row-major. Mirror the dense_with_perm transpose. */
    int* perm_rowmajor = (int*)malloc((size_t)n_rand * n_genes * sizeof(int));
    if (!perm_rowmajor) { UNPROTECT(4); error("Failed to allocate perm_table buffer"); }
    int* perm_cm = INTEGER(perm_table_r);
    for (int r = 0; r < n_rand; r++) {
        for (int j = 0; j < n_genes; j++) {
            perm_rowmajor[(size_t)r * n_genes + j] = perm_cm[(size_t)j * n_rand + r];
        }
    }

    SEXP beta_r   = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP se_r     = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP zscore_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP pvalue_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    double* beta_ptr   = REAL(beta_r);
    double* se_ptr     = REAL(se_r);
    double* zscore_ptr = REAL(zscore_r);
    double* pvalue_ptr = REAL(pvalue_r);
    size_t output_size = (size_t)n_features * n_samples;
    for (size_t i = 0; i < output_size; i++) {
        beta_ptr[i] = NA_REAL; se_ptr[i] = NA_REAL;
        zscore_ptr[i] = NA_REAL; pvalue_ptr[i] = NA_REAL;
    }

    int status = ridge_cuda_sparse(X_ptr, n_genes, n_features,
                                Y_vals_ptr, Y_row_ind_r, Y_col_ptr_r,
                                n_samples, nnz, lambda_val, n_rand, batch_size,
                                beta_ptr, se_ptr, zscore_ptr, pvalue_ptr,
                                perm_rowmajor,
                                NULL /* col_mu */, NULL /* col_sigma */);
    free(perm_rowmajor);

    SEXP result = create_result_list(beta_r, se_r, zscore_r, pvalue_r,
                                     NA_REAL, status, ridge_status_msg(status));
    UNPROTECT(8);  // 4 dgCMatrix slots + 4 output matrices
    return result;
}


/**
 * @brief R interface: sparse ridge with perm table + in-flight col-norm.
 *
 * Like ridge_cuda_sparse_with_perm_r but additionally accepts caller-
 * supplied column means (col_mu_r) and column stds (col_sigma_r) as
 * numeric vectors of length n_samples (or R NULL to skip that part of
 * the correction). Routes to the cusparseSpMM kernel which applies
 *   β = (β_raw - c⊗μ) / σ
 * inside the per-perm loop, giving correct β / SE / z / pvalue for
 * the column-normalized statistic without ever densifying Y.
 */
SEXP ridge_cuda_sparse_with_perm_norm_r(SEXP X_r, SEXP Y_r, SEXP lambda_r,
                                         SEXP n_rand_r, SEXP batch_size_r,
                                         SEXP device_id_r, SEXP perm_table_r,
                                         SEXP col_mu_r, SEXP col_sigma_r) {
    if (!isMatrix(X_r) || !isReal(X_r)) error("X must be a numeric matrix");
    if (!inherits(Y_r, "dgCMatrix")) error("Y must be a dgCMatrix object");
    if (!isReal(lambda_r) || length(lambda_r) != 1) error("lambda must be scalar numeric");
    if (!isInteger(n_rand_r) || length(n_rand_r) != 1) error("n_rand must be scalar integer");
    if (!isInteger(batch_size_r) || length(batch_size_r) != 1) error("batch_size must be scalar integer");
    if (!isInteger(device_id_r) || length(device_id_r) != 1) error("device_id must be scalar integer");
    if (!isMatrix(perm_table_r) || !isInteger(perm_table_r)) error("perm_table must be integer matrix");
    int has_mu    = !isNull(col_mu_r);
    int has_sigma = !isNull(col_sigma_r);
    if (has_mu    && !isReal(col_mu_r))    error("col_mu must be a numeric vector or NULL");
    if (has_sigma && !isReal(col_sigma_r)) error("col_sigma must be a numeric vector or NULL");

    SEXP y_vals_sexp    = PROTECT(R_do_slot(Y_r, install("x")));
    SEXP y_col_ptr_sexp = PROTECT(R_do_slot(Y_r, install("p")));
    SEXP y_row_ind_sexp = PROTECT(R_do_slot(Y_r, install("i")));
    SEXP y_dims_sexp    = PROTECT(R_do_slot(Y_r, install("Dim")));

    SEXP dim_X = getAttrib(X_r, R_DimSymbol);
    SEXP dim_P = getAttrib(perm_table_r, R_DimSymbol);
    int n_genes    = INTEGER(dim_X)[0];
    int n_features = INTEGER(dim_X)[1];
    int n_samples  = INTEGER(y_dims_sexp)[1];
    int nnz        = length(y_vals_sexp);
    if (INTEGER(y_dims_sexp)[0] != n_genes) {
        UNPROTECT(4); error("X and Y must have same number of rows");
    }
    if (has_mu    && length(col_mu_r)    != n_samples) {
        UNPROTECT(4); error("col_mu length must equal n_samples (%d)", n_samples);
    }
    if (has_sigma && length(col_sigma_r) != n_samples) {
        UNPROTECT(4); error("col_sigma length must equal n_samples (%d)", n_samples);
    }

    double lambda_val = REAL(lambda_r)[0];
    int n_rand     = INTEGER(n_rand_r)[0];
    int batch_size = INTEGER(batch_size_r)[0];
    int device_id  = asInteger(device_id_r);
    if (lambda_val < 0.0) { UNPROTECT(4); error("lambda must be non-negative"); }
    if (n_rand <= 0)      { UNPROTECT(4); error("n_rand must be positive"); }
    if (INTEGER(dim_P)[0] != n_rand)   { UNPROTECT(4); error("perm_table must have n_rand rows"); }
    if (INTEGER(dim_P)[1] != n_genes)  { UNPROTECT(4); error("perm_table must have n_genes columns"); }
    if (batch_size < 0) batch_size = 0;

    int init_status = ridge_cuda_init(device_id);
    if (init_status != 0) { UNPROTECT(4); error("CUDA init failed (%d)", init_status); }

    /* Transpose perm_table from R column-major to C row-major */
    int* perm_rowmajor = (int*)malloc((size_t)n_rand * n_genes * sizeof(int));
    if (!perm_rowmajor) { UNPROTECT(4); error("perm buffer alloc failed"); }
    int* perm_cm = INTEGER(perm_table_r);
    for (int r = 0; r < n_rand; r++) {
        for (int j = 0; j < n_genes; j++) {
            perm_rowmajor[(size_t)r * n_genes + j] = perm_cm[(size_t)j * n_rand + r];
        }
    }

    SEXP beta_r   = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP se_r     = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP zscore_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    SEXP pvalue_r = PROTECT(allocMatrix(REALSXP, n_features, n_samples));
    double* beta_ptr   = REAL(beta_r);
    double* se_ptr     = REAL(se_r);
    double* zscore_ptr = REAL(zscore_r);
    double* pvalue_ptr = REAL(pvalue_r);
    size_t output_size = (size_t)n_features * n_samples;
    for (size_t i = 0; i < output_size; i++) {
        beta_ptr[i] = NA_REAL; se_ptr[i] = NA_REAL;
        zscore_ptr[i] = NA_REAL; pvalue_ptr[i] = NA_REAL;
    }

    int status = ridge_cuda_sparse(REAL(X_r), n_genes, n_features,
                                   REAL(y_vals_sexp), INTEGER(y_row_ind_sexp), INTEGER(y_col_ptr_sexp),
                                   n_samples, nnz, lambda_val, n_rand, batch_size,
                                   beta_ptr, se_ptr, zscore_ptr, pvalue_ptr,
                                   perm_rowmajor,
                                   has_mu    ? REAL(col_mu_r)    : NULL,
                                   has_sigma ? REAL(col_sigma_r) : NULL);
    free(perm_rowmajor);

    SEXP result = create_result_list(beta_r, se_r, zscore_r, pvalue_r,
                                     NA_REAL, status, ridge_status_msg(status));
    UNPROTECT(8);
    return result;
}


/**
 * @brief Host-side Fisher-Yates permutation table using C stdlib rand().
 *
 * Mirrors RidgeFast's build_perm_table srand path so the same
 * permutation stream is consumed when rng_method="srand" is selected
 * in either package. Returns (nrand x n) integer matrix, 0-indexed,
 * column-major per R convention.
 */
/**
 * @brief In-place column z-score for a numeric matrix (R copy-on-modify
 *        bypass).
 *
 * Equivalent to:
 *   mu    <- matrixStats::colMeans2(Y)
 *   sigma <- matrixStats::colSds(Y)        # sample sd, ddof = 1
 *   sigma[sigma == 0] <- 1
 *   for (j in seq_along(mu)) Y[, j] <- (Y[, j] - mu[j]) / sigma[j]
 *
 * but operates directly on the SEXP's REAL data without R's copy-on-
 * modify firing on the first column-write — which on a 14k×100k double
 * matrix means saving an ~11 GB transient that the R-level for-loop
 * inevitably allocates the first time it touches Y[, j] (because Y's
 * NAMED >= 1 inside any function body it's passed to).
 *
 * MUTATES the matrix passed in. Only safe when the caller knows Y is
 * not shared elsewhere (typical case: freshly loaded inside a function
 * scope, e.g. Y <- h5read(...); col_zscore_inplace(Y)).
 */
SEXP col_zscore_inplace_r(SEXP Y_r) {
    if (!isMatrix(Y_r) || !isReal(Y_r))
        error("Y must be a numeric matrix");
    SEXP dim_r = getAttrib(Y_r, R_DimSymbol);
    int n = INTEGER(dim_r)[0];  // rows
    int m = INTEGER(dim_r)[1];  // cols
    if (n < 2) {
        warning("col_zscore_inplace_r: n<2, sd undefined; returning Y unchanged");
        return Y_r;
    }
    double *y = REAL(Y_r);  // R matrices are column-major: y[i + j*n]
    const double n_d = (double)n;
    const double n_minus_1 = (double)(n - 1);

    for (int j = 0; j < m; j++) {
        double *col = y + (size_t)j * (size_t)n;
        // Two-pass: mean first, then sample variance via deviations
        // (avoids catastrophic cancellation that the naïve
        // E[X²] - (E[X])² formula introduces at large n).
        double sum = 0.0;
        for (int i = 0; i < n; i++) sum += col[i];
        const double mu = sum / n_d;
        double ss = 0.0;
        for (int i = 0; i < n; i++) {
            const double d = col[i] - mu;
            ss += d * d;
        }
        double sigma = sqrt(ss / n_minus_1);
        if (sigma == 0.0) sigma = 1.0;  // match colSds convention
        const double inv_sigma = 1.0 / sigma;
        for (int i = 0; i < n; i++) {
            col[i] = (col[i] - mu) * inv_sigma;
        }
    }
    return Y_r;
}


SEXP build_srand_perm_table_r(SEXP n_r, SEXP n_rand_r, SEXP seed_r) {
    if (!isInteger(n_r)     || length(n_r)     != 1) error("n must be a single integer value");
    if (!isInteger(n_rand_r)|| length(n_rand_r)!= 1) error("n_rand must be a single integer value");
    if (!isInteger(seed_r)  || length(seed_r)  != 1) error("seed must be a single integer value");

    int n     = asInteger(n_r);
    int nrand = asInteger(n_rand_r);
    unsigned int seed = (unsigned int)asInteger(seed_r);
    if (n     < 1) error("n must be >= 1");
    if (nrand < 1) error("n_rand must be >= 1");

    SEXP result = PROTECT(allocMatrix(INTSXP, nrand, n));
    int* result_ptr = INTEGER(result);

    int* temp = (int*)malloc((size_t)n * sizeof(int));
    if (!temp) { UNPROTECT(1); error("Failed to allocate srand perm buffer"); }
    for (int k = 0; k < n; k++) temp[k] = k;

    srand(seed);
    for (int r = 0; r < nrand; r++) {
        // Fisher-Yates forward, matching RidgeFast's shuffle_srand exactly:
        //   j = i + rand() / (RAND_MAX / (n - i) + 1)
        // The division-based form (not modulo) avoids low-bit bias and is
        // what SecAct's original ridge.c uses.
        for (int i = 0; i < n - 1; i++) {
            int j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = temp[j]; temp[j] = temp[i]; temp[i] = t;
        }
        for (int j = 0; j < n; j++) {
            result_ptr[(size_t)j * nrand + r] = temp[j];
        }
    }
    free(temp);
    UNPROTECT(1);
    return result;
}


//----------------------------------------------------------------------------//
// R Package Registration                                                     //
//----------------------------------------------------------------------------//

// Define the table of callable C/C++ functions accessible from R
static const R_CallMethodDef CallEntries[] = {
    // Environment Management
    {"check_cuda_available_r",       (DL_FUNC) &check_cuda_available_r,       1},
    {"get_cuda_devices_r",           (DL_FUNC) &get_cuda_devices_r,           1},
    {"cleanup_cuda_r",               (DL_FUNC) &cleanup_cuda_r,               0},
    {"ridge_cuda_get_memory_info_r", (DL_FUNC) &ridge_cuda_get_memory_info_r, 1}, // Added

    // Core Ridge Functions - Updated with batch_size parameter
    {"ridge_cuda_dense_r",            (DL_FUNC) &ridge_cuda_dense_r,            6},
    {"ridge_cuda_dense_with_perm_r",  (DL_FUNC) &ridge_cuda_dense_with_perm_r,  7},
    {"build_srand_perm_table_r",      (DL_FUNC) &build_srand_perm_table_r,      3},
    {"col_zscore_inplace_r",          (DL_FUNC) &col_zscore_inplace_r,          1},
    {"ridge_cuda_sparse_r",           (DL_FUNC) &ridge_cuda_sparse_r,           6},
    {"ridge_cuda_sparse_with_perm_r", (DL_FUNC) &ridge_cuda_sparse_with_perm_r, 7},
    {"ridge_cuda_sparse_with_perm_norm_r", (DL_FUNC) &ridge_cuda_sparse_with_perm_norm_r, 9},

    // --- SCALING ENTRIES ---
    {"ridge_cuda_scale_dense_matrix_r",  (DL_FUNC) &ridge_cuda_scale_dense_matrix_r, 2},
    {"ridge_cuda_scale_sparse_matrix_csc_r", (DL_FUNC) &ridge_cuda_scale_sparse_matrix_csc_r, 2},
    
    // Advanced/Utility Functions - Updated with batch_size parameter
    {"ridge_cuda_memory_requirements_r", (DL_FUNC) &ridge_cuda_memory_requirements_r, 7}, // Now 7 params (+batch_size)
    {"ridge_cuda_set_memory_options_r",  (DL_FUNC) &ridge_cuda_set_memory_options_r,  3}, // Placeholder
    {"ridge_cuda_set_async_mode_r",      (DL_FUNC) &ridge_cuda_set_async_mode_r,      1}, // Placeholder

    // Terminator
    {NULL, NULL, 0}
};

// Initialization function called when the package is loaded by R
extern "C" {
    void R_init_RidgeCuda(DllInfo *dll) {
        // Register the C/C++ routines defined in CallEntries
        R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
        // Ensure R finds symbols dynamically but doesn't require all to be present (?)
        // R_useDynamicSymbols(dll, FALSE); // Standard for modern packages
        // R_forceSymbols(dll, TRUE); // Usually not needed with FALSE above
        R_useDynamicSymbols(dll, TRUE); // Required if symbols aren't explicitly registered elsewhere

    }
} // extern "C"