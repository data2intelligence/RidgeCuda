/**
 * ridge_cuda.h - NVIDIA CUDA optimized ridge regression interface
 * Supports both dense x dense and dense x sparse (CSC) operations
 * Provides permutation testing and t-test capabilities
 */
#ifndef RIDGE_CUDA_H
#define RIDGE_CUDA_H

#include <stddef.h> // For size_t

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* CUDA Environment Management                                                */
/* ========================================================================== */

/**
 * @brief Initialize CUDA environment and library handles for a specific device.
 *
 * Selects the CUDA device, creates cuBLAS, cuSOLVER, cuSPARSE, and cuRAND
 * handles. Must be called before any other ridge_cuda functions.
 * It's safe to call multiple times; it will re-initialize if the device_id
 * changes or clean up and initialize if called again for the same device.
 *
 * @param device_id The 0-based index of the CUDA device to use.
 * @return 0 on success.
 * @return Negative value on failure (e.g., -1: count failed, -2: no devices,
 *         -3: invalid ID, -4: set device failed, -5..-9: handle creation failed).
 */
int ridge_cuda_init(int device_id);

/**
 * @brief Release all CUDA resources and reset the device.
 *
 * Destroys library handles and calls cudaDeviceReset(). Should be called when
 * CUDA operations are finished to free resources. Automatically called by
 * ridge_cuda_init() if re-initializing.
 */
void ridge_cuda_cleanup(void);

/**
 * @brief Get information about available CUDA devices.
 *
 * @param device_count Output pointer to store the number of detected devices.
 * @param device_names (Optional) Array of char pointers to store device names.
 *                     The caller must allocate memory for each name string.
 *                     Can be NULL if names are not needed.
 * @param max_name_len The maximum length (including null terminator) of each
 *                     string allocated in `device_names`. Ignored if `device_names` is NULL.
 * @param device_memories (Optional) Array to store total global memory size for each
 *                        device in bytes. Can be NULL if memory info is not needed.
 * @return 0 on success.
 * @return -1 if getting device count fails.
 * @return -2 if getting device properties fails for any device.
 */
int ridge_cuda_get_devices(int* device_count, char** device_names,
                          int max_name_len, size_t* device_memories);

/**
 * @brief Get CUDA memory usage information for the currently selected device.
 *
 * Note: ridge_cuda_init() must have been called successfully first.
 *
 * @param free_memory Output pointer to store free memory in bytes.
 * @param total_memory Output pointer to store total memory in bytes.
 * @return 0 on success.
 * @return Non-zero if cudaMemGetInfo fails.
 */
int ridge_cuda_get_memory_info(size_t* free_memory, size_t* total_memory);


/* ========================================================================== */
/* Ridge Regression Implementations                                           */
/* ========================================================================== */

/**
 * @brief Perform ridge regression with dense X and dense Y matrices using CUDA.
 *
 * Calculates beta = (X'X + lambda*I)^-1 * X'Y.
 * Optionally performs a t-test (n_rand = 0) or permutation test (n_rand > 0)
 * for significance.
 *
 * @param X Input dense matrix X (n_genes x n_features), stored row-major (C-style).
 * @param Y Input dense matrix Y (n_genes x n_samples), stored row-major (C-style).
 * @param n_genes Number of rows in X and Y (observations).
 * @param n_features Number of columns in X (predictors).
 * @param n_samples Number of columns in Y (responses/tasks).
 * @param lambda_val Ridge regularization parameter (lambda >= 0).
 * @param n_rand Number of permutations for significance testing.
 *               If n_rand = 0, performs a t-test.
 *               If n_rand > 0, performs a permutation test.
 * @param batch_size Number of Y columns to process in each batch (default=0 means all columns).
 *                  Reduces memory usage by processing Y in smaller batches.
 * @param beta Output buffer for beta coefficients (n_features x n_samples), row-major.
 *             Caller must allocate memory.
 * @param se Output buffer for standard errors (n_features x n_samples), row-major.
 *           Caller must allocate memory.
 * @param zscore Output buffer for z-scores (permutation test) or t-statistics (t-test),
 *               (n_features x n_samples), row-major. Caller must allocate memory.
 * @param pvalue Output buffer for p-values (n_features x n_samples), row-major.
 *               Caller must allocate memory.
 * @return 0 on success.
 * @return Non-zero error code on failure (negative for validation/init errors,
 *         positive for CUDA/library errors during computation).
 */
int ridge_cuda_dense(
    const double *X,
    const double *Y,
    int n_genes,
    int n_features,
    int n_samples,
    double lambda_val,
    int n_rand,
    int batch_size,
    double *beta,
    double *se,
    double *zscore,
    double *pvalue,
    const int *perm_table  /* Optional: n_rand x n_genes, row-major.
                              NULL = fisher_yates (platform rand, not
                              reproducible). Non-NULL = use provided
                              permutations (e.g. MT19937 seed 0 from R)
                              for cross-backend bitwise reproducibility. */
);

/**
 * @brief Perform ridge regression with dense X and sparse Y (CSC format) using CUDA.
 *
 * Calculates beta = (X'X + lambda*I)^-1 * X'Y.
 * Performs a permutation test (n_rand > 0) for significance.
 * Uses 0-based indexing for sparse matrix representation (Compressed Sparse Column - CSC).
 *
 * @param X Input dense matrix X (n_genes x n_features), stored row-major (C-style).
 * @param n_genes Number of rows in X and Y (observations).
 * @param n_features Number of columns in X (predictors).
 * @param Y_vals Array of non-zero values of Y (length nnz), from CSC format.
 * @param Y_row_indices Array of row indices for non-zero values (length nnz), from CSC format, 0-based.
 * @param Y_col_pointers Array of column pointers (length n_samples + 1), from CSC format, 0-based.
 * @param n_samples Number of columns in Y (responses/tasks).
 * @param nnz Number of non-zero elements in Y.
 * @param lambda_val Ridge regularization parameter (lambda >= 0).
 * @param n_rand Number of permutations for significance testing.
 *               If n_rand > 0, performs a permutation test.
 * @param batch_size Number of Y columns to process in each batch (default=0 means all columns).
 *                  Reduces memory usage by processing Y in smaller batches.
 * @param beta Output buffer for beta coefficients (n_features x n_samples), row-major.
 *             Caller must allocate memory.
 * @param se Output buffer for standard errors (n_features x n_samples), row-major.
 *           Caller must allocate memory.
 * @param zscore Output buffer for z-scores (permutation test),
 *               (n_features x n_samples), row-major. Caller must allocate memory.
 * @param pvalue Output buffer for p-values (n_features x n_samples), row-major.
 *               Caller must allocate memory.
 * @return 0 on success.
 * @return Non-zero error code on failure (negative for validation/init errors,
 *         positive for CUDA/library errors during computation).
 */
int ridge_cuda_sparse(
    const double *X,
    int n_genes,
    int n_features,
    const double *Y_vals,
    const int    *Y_row_indices,
    const int    *Y_col_pointers,
    int n_samples,
    int nnz,
    double lambda_val,
    int n_rand,
    int batch_size,
    double *beta,
    double *se,
    double *zscore,
    double *pvalue
);

/**
 * @brief Calculate estimated GPU memory requirements for a ridge regression task.
 *
 * Provides an estimate based on matrix dimensions and operation mode.
 * Takes into account batch processing to provide accurate memory estimates.
 *
 * @param n_genes Number of genes (rows in X and Y).
 * @param n_features Number of features (columns in X).
 * @param n_samples Number of samples (columns in Y).
 * @param nnz Number of non-zero elements in Y (only used if is_sparse = 1).
 * @param is_sparse Flag indicating if Y is sparse (1) or dense (0).
 * @param n_rand Number of permutations (0 for t-test, >0 for permutation test).
 * @param batch_size Number of Y columns to process in each batch (0 means all columns).
 * @return Estimated memory requirement in bytes. Returns 0 if inputs are invalid.
 */
size_t ridge_cuda_memory_requirements(
    int n_genes,
    int n_features,
    int n_samples,
    int nnz,
    int is_sparse,
    int n_rand,
    int batch_size
);



/**
 * @brief Set memory management options for CUDA operations.
 *
 * Configures memory pooling behavior which can improve performance for
 * repeated allocations of similar size by reducing fragmentation and
 * system calls. When enabled, the CUDA driver maintains a pool of memory
 * allocations that can be reused.
 *
 * @param enable_pool Whether to enable memory pooling (0 = disabled, 1 = enabled).
 * @param allocation_size Initial size of memory pool in bytes (0 = use CUDA default).
 * @param release_threshold Threshold for memory release back to system in bytes
 *                         (0 = use CUDA default).
 * @return 0 on success, non-zero error code otherwise.
 */
int ridge_cuda_set_memory_options(int enable_pool,
                                 size_t allocation_size,
                                 size_t release_threshold);


/**
 * @brief Set asynchronous execution mode for CUDA operations.
 *
 * When enabled, some CUDA operations may execute asynchronously, potentially
 * improving performance by overlapping computation and data transfer.
 * This setting primarily affects how synchronization is handled within
 * the ridge_cuda implementations.
 *
 * @param enable_async Whether to enable asynchronous execution (0 = disabled/synchronous, 
 *                    1 = enabled/asynchronous).
 * @return The previous mode setting (0 = disabled, 1 = enabled).
 */
int ridge_cuda_set_async_mode(int enable_async);


/* ========================================================================== */
/* Data Scaling Functions (New)                                               */
/* ========================================================================== */

/**
 * @brief Performs column-wise standard scaling (Z-score) on a dense matrix on the GPU.
 *
 * Scales the matrix `d_matrix` in-place: `d_matrix[i,j] = (d_matrix[i,j] - mean[j]) / sd[j]`.
 * Columns with standard deviation close to zero will have their elements set to zero.
 * Assumes `d_matrix` is column-major.
 *
 * @param d_matrix Device pointer to the double-precision matrix data (column-major).
 * @param n_rows Number of rows in the matrix.
 * @param n_cols Number of columns in the matrix.
 * @param d_means Device pointer to store the calculated mean for each column (size n_cols).
 *                Must be allocated by the caller.
 * @param d_sds Device pointer to store the calculated standard deviation for each column (size n_cols).
 *              Must be allocated by the caller.
 * @return 0 on success, non-zero on CUDA error.
 */
int ridge_cuda_scale_dense_matrix(
    double *d_matrix,
    int n_rows,
    int n_cols,
    double *d_means,
    double *d_sds
);

/**
 * @brief Performs column-wise sparse-aware scaling on a sparse matrix (CSC format) on the GPU.
 *
 * Calculates mean and standard deviation for each column using *only* the non-zero values.
 * Scales *only* the non-zero values in `d_vals` in-place using the calculated non-zero mean/SD.
 * Non-zero values in columns with zero non-zero SD will be set to zero.
 * Zero elements in the conceptual matrix remain zero.
 * Assumes 0-based indexing for CSC arrays.
 *
 * @param d_vals Device pointer to the non-zero values (size nnz). Modified in-place.
 * @param d_row_indices Device pointer to the row indices for non-zero values (size nnz).
 * @param d_col_pointers Device pointer to the column pointers (size n_cols + 1).
 * @param n_rows Number of rows in the conceptual sparse matrix.
 * @param n_cols Number of columns in the conceptual sparse matrix.
 * @param nnz Number of non-zero elements.
 * @param d_means_nz Device pointer to store the calculated mean of non-zeros for each column (size n_cols).
 *                   Must be allocated by the caller.
 * @param d_sds_nz Device pointer to store the calculated SD of non-zeros for each column (size n_cols).
 *                 Must be allocated by the caller.
 * @param d_counts_nz Device pointer to store the count of non-zeros per column (optional, can be NULL).
 *                    If not NULL, must be allocated by the caller (size n_cols).
 * @return 0 on success, non-zero on CUDA error.
 */
int ridge_cuda_scale_sparse_matrix_csc(
    double *d_vals,
    const int *d_row_indices, // const as they are not modified
    const int *d_col_pointers, // const as they are not modified
    int n_rows,
    int n_cols,
    int nnz,
    double *d_means_nz,
    double *d_sds_nz,
    int *d_counts_nz // Optional output
);


#ifdef __cplusplus
} // extern "C"
#endif

#endif /* RIDGE_CUDA_H */