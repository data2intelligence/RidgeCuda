/**
 * ridge_cuda.cu - NVIDIA CUDA optimized ridge regression
 * Supports dense x dense and dense x sparse (CSC) operations
 * Provides permutation testing capabilities. (T-test functionality removed)
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h> // For getpid()

// CUDA Runtime
#include <cuda_runtime.h>
// CUDA Libraries
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <curand.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h> // For thrust::device

// Include our header
#include "ridge_cuda.h"

// Constants
#define EPS 1e-12
#define MAX_GPU_NAME 256
#define SCALE_BLOCK_SIZE 256
#define MIN_SD_THRESHOLD 1e-10 // Threshold below which SD is considered zero

// Global handles & state
static cublasHandle_t cublas_handle = NULL;
static cusolverDnHandle_t cusolver_handle = NULL;
static cusparseHandle_t cusparse_handle = NULL;
static curandGenerator_t rand_generator = NULL;
static int cuda_initialized = 0;
static int current_device = -1;
static int memory_pool_enabled = 0;
static int async_mode_enabled = 0;

// Error handling macros
#define CHECK_CUDA(call) { \
    cudaError_t cuda_status = call; \
    if (cuda_status != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(cuda_status), __FILE__, __LINE__); \
        return 1; \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t cublas_status = call; \
    if (cublas_status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error Status: %d at %s:%d\n", \
                (int)cublas_status, __FILE__, __LINE__); \
        return 2; \
    } \
}

#define CHECK_CUSOLVER(call) { \
    cusolverStatus_t cusolver_status = call; \
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER Error Status: %d at %s:%d\n", \
                (int)cusolver_status, __FILE__, __LINE__); \
        return 3; \
    } \
}

#define CHECK_CUSPARSE(call) { \
    cusparseStatus_t cusparse_status = call; \
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE Error Status: %d at %s:%d\n", \
                (int)cusparse_status, __FILE__, __LINE__); \
        return 4; \
    } \
}

#define CHECK_CURAND(call) { \
    curandStatus_t curand_status = call; \
    if (curand_status != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "cuRAND Error Status: %d at %s:%d\n", \
                (int)curand_status, __FILE__, __LINE__); \
        return 5; \
    } \
}

// --- Forward declarations ---
// Kernels needed for Core + Permutation:
__global__ void addDiagonalConstant(double *matrix, double value, int n);
__global__ void calculateAbsValues(const double *input, double *output, size_t n);
__global__ void permuteColumnsKernel(const double *input, double *output, const int *indices, int rows, int cols);
__global__ void permuteRowsKernel(const double *input, double *output, const int *indices, int rows, int cols);
__global__ void updatePermutationStats(const double *beta_perm, const double *abs_beta_obs, double *sum_b, double *sum_b2, double *count_ge, size_t n, double eps);

// Host helpers needed for Core + Permutation:
static void permute_T_columns_cuda(double *d_T_permuted, double *d_T, int *d_indices, int p, int n);
static void permute_T_transpose_rows_cuda(double *d_Tt_permuted, double *d_Tt, int *d_indices, int n, int p);
static void fisher_yates_shuffle(int *arr, int n);

//----------------------------------------------------------------------------//
// KERNEL IMPLEMENTATIONS                                                     //
//----------------------------------------------------------------------------//

__global__ void addDiagonalConstant(double *matrix, double value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        matrix[idx * n + idx] += value;
    }
}

__global__ void calculateAbsValues(const double *input, double *output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fabs(input[idx]);
    }
}

__global__ void permuteColumnsKernel(const double *input, double *output,
                                    const int *indices, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Assuming input/output are column-major (Fortran style as often used with BLAS/LAPACK)
    // Access element at (row, col) as input[col * rows + row]
    if (row < rows && col < cols) {
        int source_col = indices[col]; // Permute based on column index
        if (source_col >= 0 && source_col < cols) {
             output[col * rows + row] = input[source_col * rows + row];
        } else {
             output[col * rows + row] = 0.0; // Handle potential out-of-bounds if needed
        }
    }
     // If row-major, use: output[row * cols + col] = input[row * cols + source_col];
}


__global__ void permuteRowsKernel(const double *input, double *output, const int *indices,
                             int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Assuming input/output are column-major
    // Access element at (row, col) as input[col * rows + row]
    if (row < rows && col < cols) {
        int source_row = indices[row]; // Permute based on row index
        if (source_row >= 0 && source_row < rows) {
             output[col * rows + row] = input[col * rows + source_row];
        } else {
            output[col * rows + row] = 0.0; // Handle potential out-of-bounds if needed
        }
    }
    // If row-major, use: output[row * cols + col] = input[source_row * cols + col];
}


__global__ void updatePermutationStats(const double *beta_perm, const double *abs_beta_obs,
                                      double *sum_b, double *sum_b2, double *count_ge,
                                      size_t n, double eps) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = beta_perm[idx];
        double abs_val = fabs(val);
        double abs_obs = abs_beta_obs[idx];
        atomicAdd(&sum_b[idx], val);
        atomicAdd(&sum_b2[idx], val * val);
        if (abs_val >= abs_obs - eps) {
            atomicAdd(&count_ge[idx], 1.0);
        }
    }
}

// --- Kernel Helper Structures (Optional but can be useful) ---
// Example: Functor for reduction
// struct SumSqCount {
//     double sum;
//     double sum_sq;
//     int count;
//     __host__ __device__ SumSqCount() : sum(0.0), sum_sq(0.0), count(0) {}
// };
// (Thrust might handle this more easily for simple sums/counts)


// --- DENSE SCALING IMPLEMENTATION ---

// Kernel to calculate sum and sum_sq per column (using atomics for simplicity, can be optimized with shared memory reduction)
__global__ void calculateColSumsDenseKernel(
    const double *matrix, // n_rows x n_cols (col-major)
    int n_rows,
    int n_cols,
    double *d_sums,     // size n_cols
    double *d_sums_sq   // size n_cols
) {
    int col = blockIdx.x; // Each block processes one column

    if (col >= n_cols) return;

    double local_sum = 0.0;
    double local_sum_sq = 0.0;

    // Iterate over rows in this column using grid-stride loop within the block
    for (int row = threadIdx.x; row < n_rows; row += blockDim.x) {
        double val = matrix[col * n_rows + row];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Simple reduction using shared memory (more efficient than pure atomics)
    __shared__ double s_sum[SCALE_BLOCK_SIZE];
    __shared__ double s_sum_sq[SCALE_BLOCK_SIZE];

    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    // Reduce in shared memory (example: power-of-2 reduction)
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + offset];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result for the column
    if (threadIdx.x == 0) {
        d_sums[col] = s_sum[0];
        d_sums_sq[col] = s_sum_sq[0];
    }
}


// Kernel to apply scaling using pre-calculated means and SDs
__global__ void applyColScalingDenseKernel(
    double *matrix, // n_rows x n_cols (col-major) - Modified in-place
    int n_rows,
    int n_cols,
    const double *d_means, // size n_cols
    const double *d_sds    // size n_cols
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_rows && col < n_cols) {
        size_t idx = (size_t)col * n_rows + row;
        double mean = d_means[col];
        double sd = d_sds[col];

        if (sd > MIN_SD_THRESHOLD) {
            matrix[idx] = (matrix[idx] - mean) / sd;
        } else {
            matrix[idx] = 0.0; // Set to 0 if SD is too small
        }
    }
}

// --- SPARSE SCALING IMPLEMENTATION (CSC) ---

// Kernel to calculate sum, sum_sq, and count of NON-ZERO elements per column
__global__ void calculateColSumsSparseNzKernel(
    const double *d_vals,          // size nnz
    const int *d_col_pointers, // size n_cols + 1
    int n_cols,
    double *d_sums_nz,     // size n_cols
    double *d_sums_sq_nz,  // size n_cols
    int *d_counts_nz     // size n_cols
) {
    int col = blockIdx.x; // Each block processes one column
    if (col >= n_cols) return;

    double local_sum = 0.0;
    double local_sum_sq = 0.0;
    int local_count = 0;

    int start_idx = d_col_pointers[col];
    int end_idx = d_col_pointers[col + 1];

    // Iterate over non-zeros in this column using grid-stride loop within the block
    for (int i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
        double val = d_vals[i];
        local_sum += val;
        local_sum_sq += val * val;
        local_count++;
    }

    // Reduce within the block using shared memory
    __shared__ double s_sum[SCALE_BLOCK_SIZE];
    __shared__ double s_sum_sq[SCALE_BLOCK_SIZE];
    __shared__ int s_count[SCALE_BLOCK_SIZE];

    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    s_count[threadIdx.x] = local_count;
    __syncthreads();

    // Power-of-2 reduction in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + offset];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + offset];
            s_count[threadIdx.x] += s_count[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result for the column
    if (threadIdx.x == 0) {
        d_sums_nz[col] = s_sum[0];
        d_sums_sq_nz[col] = s_sum_sq[0];
        d_counts_nz[col] = s_count[0];
    }
}


// Kernel to apply sparse scaling (modifies d_vals in-place)
__global__ void applyColScalingSparseNzKernel(
    double *d_vals,              // size nnz - Modified in-place
    const int *d_col_pointers,     // size n_cols + 1
    int n_cols,
    const double *d_means_nz,    // size n_cols
    const double *d_sds_nz       // size n_cols
) {
    int col = blockIdx.x; // Each block processes one column
    if (col >= n_cols) return;

    double mean_nz = d_means_nz[col];
    double sd_nz = d_sds_nz[col];
    bool scale_col = (sd_nz > MIN_SD_THRESHOLD); // Check if SD is valid for scaling

    int start_idx = d_col_pointers[col];
    int end_idx = d_col_pointers[col + 1];

    // Iterate over non-zeros in this column using grid-stride loop
    for (int i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
        if (scale_col) {
            d_vals[i] = (d_vals[i] - mean_nz) / sd_nz;
        } else {
            d_vals[i] = 0.0; // Set non-zero value to 0 if SD is invalid
        }
    }
}


//----------------------------------------------------------------------------//
// HOST HELPER FUNCTIONS                                                      //
//----------------------------------------------------------------------------//

static void fisher_yates_shuffle(int *arr, int n) {
    if (n <= 1) return;
    // Seed rand() once if needed, though often seeded elsewhere
    // srand(time(NULL)); // Or use a better RNG if available
    for (int i = n - 1; i > 0; i--) {
        int k = rand() % (i + 1);
        int temp = arr[i];
        arr[i] = arr[k];
        arr[k] = temp;
    }
}

// Note: Assuming column-major layout for matrices passed to BLAS/LAPACK
// T is (p x n), T_permuted is (p x n)
static void permute_T_columns_cuda(double *d_T_permuted, double *d_T,
                                  int *d_indices, int p /*rows*/, int n /*cols*/) {
    dim3 threadsPerBlock(16, 16);
    // Grid dimensions based on matrix size (p rows, n cols)
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (p + threadsPerBlock.y - 1) / threadsPerBlock.y);
    permuteColumnsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_T, d_T_permuted, d_indices, p, n);
    // Add error checking after kernel launch for debugging
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after launching permuteColumnsKernel: %s\n", cudaGetErrorString(err));
        // Consider how to propagate this error if needed
    }
    // No need to synchronize here usually, subsequent CUDA calls will implicitly sync if needed
}

// Note: Assuming column-major layout
// Tt is (n x p), Tt_permuted is (n x p)
static void permute_T_transpose_rows_cuda(double *d_Tt_permuted, double *d_Tt,
                                          int *d_indices, int n /*rows*/, int p /*cols*/) {
    dim3 threadsPerBlock(16, 16);
    // Grid dimensions based on matrix size (n rows, p cols)
    dim3 blocksPerGrid((p + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    permuteRowsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_Tt, d_Tt_permuted, d_indices, n, p);
    // Add error checking after kernel launch for debugging
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after launching permuteRowsKernel: %s\n", cudaGetErrorString(err));
        // Consider how to propagate this error if needed
    }
     // No need to synchronize here usually
}


extern "C" {

//----------------------------------------------------------------------------//
// UTILITY & ENVIRONMENT FUNCTIONS                                            //
//----------------------------------------------------------------------------//

int ridge_cuda_init(int device_id) {
    int device_count;
    cudaError_t err;
    if (cuda_initialized && current_device == device_id) return 0;
    if (cuda_initialized) ridge_cuda_cleanup();
    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) { fprintf(stderr, "Failed to get CUDA device count: %s\n", cudaGetErrorString(err)); return -1; }
    if (device_count == 0) { fprintf(stderr, "No CUDA-capable devices found.\n"); return -2; }
    if (device_id < 0 || device_id >= device_count) { fprintf(stderr, "Invalid device ID %d requested. Found %d devices (0 to %d).\n", device_id, device_count, device_count - 1); return -3; }
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) { fprintf(stderr, "Failed to set CUDA device %d: %s\n", device_id, cudaGetErrorString(err)); return -4; }
    current_device = device_id;
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "Failed to create cuBLAS handle (Error %d)\n", (int)cublas_status); cudaDeviceReset(); return -5; }
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolver_handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) { fprintf(stderr, "Failed to create cuSOLVER handle (Error %d)\n", (int)cusolver_status); cublasDestroy(cublas_handle); cudaDeviceReset(); return -6; }
    cusparseStatus_t cusparse_status = cusparseCreate(&cusparse_handle);
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { fprintf(stderr, "Failed to create cuSPARSE handle (Error %d)\n", (int)cusparse_status); cusolverDnDestroy(cusolver_handle); cublasDestroy(cublas_handle); cudaDeviceReset(); return -7; }
    curandStatus_t curand_status = curandCreateGenerator(&rand_generator, CURAND_RNG_PSEUDO_DEFAULT);
    if (curand_status != CURAND_STATUS_SUCCESS) { fprintf(stderr, "Failed to create cuRAND generator (Error %d)\n", (int)curand_status); cusparseDestroy(cusparse_handle); cusolverDnDestroy(cusolver_handle); cublasDestroy(cublas_handle); cudaDeviceReset(); return -8; }
    // Seed with a combination of time and device ID for better uniqueness across multiple GPUs
    curand_status = curandSetPseudoRandomGeneratorSeed(rand_generator, (unsigned long long)time(NULL) ^ (unsigned long long)getpid() ^ (unsigned long long)device_id);
    if (curand_status != CURAND_STATUS_SUCCESS) { fprintf(stderr, "Failed to set cuRAND seed (Error %d)\n", (int)curand_status); curandDestroyGenerator(rand_generator); cusparseDestroy(cusparse_handle); cusolverDnDestroy(cusolver_handle); cublasDestroy(cublas_handle); cudaDeviceReset(); return -9; }
    cuda_initialized = 1;
    return 0;
}

void ridge_cuda_cleanup(void) {
    if (!cuda_initialized) return;
    if (rand_generator) curandDestroyGenerator(rand_generator);
    if (cusparse_handle) cusparseDestroy(cusparse_handle);
    if (cusolver_handle) cusolverDnDestroy(cusolver_handle);
    if (cublas_handle) cublasDestroy(cublas_handle);
    // Only reset the device if we successfully set one
    if (current_device >= 0) cudaDeviceReset();
    rand_generator = NULL; cusparse_handle = NULL; cusolver_handle = NULL; cublas_handle = NULL;
    cuda_initialized = 0; current_device = -1;
}

int ridge_cuda_get_devices(int* device_count, char** device_names,
                         int max_name_len, size_t* device_memories) {
    cudaError_t err;
    err = cudaGetDeviceCount(device_count);
    if (err != cudaSuccess) { fprintf(stderr, "Failed to get CUDA device count: %s\n", cudaGetErrorString(err)); *device_count = 0; return -1; }
    if (*device_count == 0) return 0;
    if (device_names == NULL && device_memories == NULL) return 0; // Nothing to fill

    for (int i = 0; i < *device_count; i++) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) { fprintf(stderr, "Failed to get properties for device %d: %s\n", i, cudaGetErrorString(err)); return -2; }
        if (device_names != NULL && device_names[i] != NULL && max_name_len > 0) {
            strncpy(device_names[i], prop.name, max_name_len - 1);
            device_names[i][max_name_len - 1] = '\0'; // Ensure null termination
        }
        if (device_memories != NULL) {
            device_memories[i] = prop.totalGlobalMem;
        }
    }
    return 0;
}


int ridge_cuda_get_memory_info(size_t* free_memory, size_t* total_memory) {
    if (!cuda_initialized || current_device < 0) {
        fprintf(stderr, "Error: CUDA not initialized or no device set. Call ridge_cuda_init() first.\n");
        if(free_memory) *free_memory = 0; if(total_memory) *total_memory = 0; return -10;
    }
    if (free_memory == NULL || total_memory == NULL) { fprintf(stderr, "Error: NULL pointer provided for memory info output.\n"); return -13; }
    cudaError_t cuda_status = cudaMemGetInfo(free_memory, total_memory);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "CUDA Error getting memory info: %s\n", cudaGetErrorString(cuda_status));
        *free_memory = 0; *total_memory = 0; return -20; // Use a different error code
    }
    return 0;
}

// --- Assuming Column-Major layout for BLAS/LAPACK ---
// X is (n_genes x n_features), p = n_features, n = n_genes
// Y is (n_genes x n_samples), n = n_genes, m = n_samples
// beta is (n_features x n_samples), p = n_features, m = n_samples
size_t ridge_cuda_memory_requirements(int n_genes, int n_features, int n_samples,
                                   int nnz, int is_sparse, int n_rand, int batch_size) {
    // --- Validation ---
    if (n_genes <= 0 || n_features <= 0 || n_samples <= 0 || n_rand < 0) {
         fprintf(stderr, "Warning: Invalid dimensions/n_rand passed to memory_requirements (%d, %d, %d, %d, %d).\n", 
                n_genes, n_features, n_samples, nnz, n_rand);
         return 0;
    }
    if (is_sparse && nnz <= 0) {
        fprintf(stderr, "Warning: is_sparse=1 but nnz=%d in memory_requirements. Assuming minimal sparse structure.\n", nnz);
        nnz = 1; // Avoid division by zero or issues with zero nnz
    }
    
    // Process batch_size: If invalid (<=0 or >n_samples), use all samples
    if (batch_size <= 0 || batch_size > n_samples) {
        batch_size = n_samples; // Process all at once
    }

    // --- Define dimensions shorthand ---
    size_t dbl_size = sizeof(double);
    size_t int_size = sizeof(int);
    size_t p = n_features;
    size_t n = n_genes;
    size_t m = n_samples;
    size_t m_batch = batch_size; // Max samples per batch
    size_t total_bytes = 0;

    // --- Common allocations for both dense and sparse (independent of batch size) ---
    // X matrices
    total_bytes += n * p * dbl_size;         // d_X (n x p)
    total_bytes += p * p * dbl_size;         // d_XtX (p x p)
    total_bytes += p * n * dbl_size;         // d_X_transpose (p x n)
    total_bytes += p * n * dbl_size;         // d_T (p x n)
    total_bytes += int_size;                 // d_info
    total_bytes += n * int_size;             // d_indices (size n) for permutation

    // Workspace estimate for Cholesky solve (potrf + potrs)
    size_t workspace_estimate = fmax(p * p, p * n) * dbl_size;
    // Buffer for cuBLAS operations
    workspace_estimate += fmax(n*p, p*m_batch) * dbl_size / 2; // Heuristic
    total_bytes += workspace_estimate;

    // --- Allocations based on batch size ---
    if (is_sparse) {
        // Full sparse Y structure
        total_bytes += nnz * dbl_size;             // d_Y_vals
        total_bytes += nnz * int_size;             // d_Y_row_indices
        total_bytes += (m + 1) * int_size;         // d_Y_col_pointers
        
        // Batch-specific matrices
        total_bytes += n * p * dbl_size;           // d_T_transpose (n x p)
        total_bytes += m_batch * p * dbl_size;     // d_beta_transpose (batch_size x p)
        total_bytes += p * m_batch * dbl_size;     // d_beta_batch (p x batch_size)
        
        // Permutation batch-specific
        if (n_rand > 0) {
            total_bytes += n * p * dbl_size;          // d_T_transpose_perm (n x p)
            total_bytes += p * m_batch * dbl_size;    // d_beta_perm (p x batch_size)
            total_bytes += m_batch * p * dbl_size;    // d_beta_perm_transpose (batch_size x p)
            total_bytes += p * m_batch * dbl_size;    // d_abs_beta_obs (p x batch_size)
            total_bytes += p * m_batch * dbl_size * 3; // d_sum_b, d_sum_b2, d_count_ge (p x batch_size each)
            
            // SpMM buffer (varies widely, estimated)
            total_bytes += nnz * dbl_size / 2 + m_batch * p * dbl_size / 4; // Heuristic
        }
    } else {
        // Batch-specific matrices for dense Y
        total_bytes += n * m_batch * dbl_size;     // d_Y_batch (n x batch_size)
        total_bytes += p * m_batch * dbl_size;     // d_beta_batch (p x batch_size)
        
        // Permutation batch-specific
        if (n_rand > 0) {
            total_bytes += p * n * dbl_size;          // d_T_permuted (p x n)
            total_bytes += p * m_batch * dbl_size;    // d_beta_perm (p x batch_size)
            total_bytes += p * m_batch * dbl_size;    // d_abs_beta_obs (p x batch_size)
            total_bytes += p * m_batch * dbl_size * 3; // d_sum_b, d_sum_b2, d_count_ge (p x batch_size each)
        }
    }

    // --- Host Memory for Result Accumulation ---
    // Full size result matrices (n_features x n_samples)
    if (n_rand > 0) {
        total_bytes += p * m * dbl_size * 3; // h_sum_b, h_sum_b2, h_count_ge (host accumulation)
    }

    // Add a fixed buffer for general overhead, library state, fragmentation etc.
    total_bytes += 64 * 1024 * 1024; // Add fixed 64MB general buffer

    return total_bytes;
}


/**
 * @brief Set memory management options for CUDA operations.
 *
 * Implements memory pooling for CUDA allocations which can improve performance,
 * especially for repeated allocations of similar sizes.
 */
int ridge_cuda_set_memory_options(int enable_pool, size_t allocation_size, size_t release_threshold) {
    if (!cuda_initialized) {
        fprintf(stderr, "Error: CUDA not initialized. Call ridge_cuda_init() first.\n");
        return -1;
    }
    
    cudaError_t err;
    
    // Store previous setting to return change status
    int prev_setting = memory_pool_enabled;
    
    // Update global state
    memory_pool_enabled = enable_pool;
    
    if (enable_pool) {
        // Set memory pool attributes
        // Use cudaDeviceSetLimit for memory management which works across CUDA versions
        err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 
                              allocation_size > 0 ? allocation_size : 128 * 1024 * 1024);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error setting malloc heap size: %s\n", cudaGetErrorString(err));
            return -2;
        }
    } else {
        // Disable memory pool - for simplicity we'll set a smaller heap limit
        err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8 * 1024 * 1024); // 8MB default
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error resetting malloc heap size: %s\n", cudaGetErrorString(err));
            return -3;
        }
    }
    
    return prev_setting; // Return previous state
}

/**
 * @brief Set asynchronous execution mode for CUDA operations.
 *
 * Controls whether operations are performed synchronously or asynchronously.
 * In asynchronous mode, functions may return before the operation is complete.
 */
int ridge_cuda_set_async_mode(int enable_async) {
    if (!cuda_initialized) {
        fprintf(stderr, "Error: CUDA not initialized. Call ridge_cuda_init() first.\n");
        return -1;
    }
    
    // Store previous setting
    int prev_setting = async_mode_enabled;
    
    // Update global state
    async_mode_enabled = enable_async;
    
    return prev_setting; // Return previous state
}

// Host function for dense scaling
extern "C" int ridge_cuda_scale_dense_matrix(
    double *d_matrix,
    int n_rows,
    int n_cols,
    double *d_means,
    double *d_sds
) {
    if (!cuda_initialized) return -10; // Or appropriate error
    if (!d_matrix || !d_means || !d_sds || n_rows <= 0 || n_cols <= 0) return -13;

    double *d_sums = NULL;
    double *d_sums_sq = NULL;

    // Allocate temporary memory for sums and sums_sq
    CHECK_CUDA(cudaMalloc((void**)&d_sums, n_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_sums_sq, n_cols * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_sums, 0, n_cols * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_sums_sq, 0, n_cols * sizeof(double)));

    // --- Calculate Sums and Sums_sq ---
    dim3 blockDimSums(SCALE_BLOCK_SIZE);
    dim3 gridDimSums(n_cols); // One block per column
    calculateColSumsDenseKernel<<<gridDimSums, blockDimSums>>>(d_matrix, n_rows, n_cols, d_sums, d_sums_sq);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for sums kernel

    // --- Calculate Mean and SD on Host (simpler) or Device (more parallel) ---
    // Let's do it on Host for simplicity here, requires D->H transfer
    double *h_sums = (double*)malloc(n_cols * sizeof(double));
    double *h_sums_sq = (double*)malloc(n_cols * sizeof(double));
    if (!h_sums || !h_sums_sq) {
        cudaFree(d_sums); cudaFree(d_sums_sq); free(h_sums); free(h_sums_sq);
        fprintf(stderr, "Host allocation failed in scale_dense\n"); return -1;
    }
    CHECK_CUDA(cudaMemcpy(h_sums, d_sums, n_cols * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sums_sq, d_sums_sq, n_cols * sizeof(double), cudaMemcpyDeviceToHost));

    double *h_means = (double*)malloc(n_cols * sizeof(double));
    double *h_sds = (double*)malloc(n_cols * sizeof(double));
     if (!h_means || !h_sds) {
         cudaFree(d_sums); cudaFree(d_sums_sq); free(h_sums); free(h_sums_sq); free(h_means); free(h_sds);
         fprintf(stderr, "Host allocation failed for mean/sd in scale_dense\n"); return -1;
     }

    double n_rows_d = (double)n_rows;
    for (int j = 0; j < n_cols; ++j) {
        h_means[j] = h_sums[j] / n_rows_d;
        double variance = (h_sums_sq[j] / n_rows_d) - (h_means[j] * h_means[j]);
        // Apply Bessel's correction if needed (n/(n-1))? Standard scale() uses n-1.
        // Let's use population variance/SD for consistency with potential direct comparison
        // Variance = E[X^2] - (E[X])^2
        if (variance < 0.0 && variance > -EPS) variance = 0.0; // Handle small float errors
        if (variance < 0.0) {
            fprintf(stderr, "Warning: Negative variance (%.4e) encountered in dense scaling col %d. Clamping to 0.\n", variance, j);
            variance = 0.0;
        }
        h_sds[j] = sqrt(variance);
    }

    // Copy calculated means and SDs to the output device pointers
    CHECK_CUDA(cudaMemcpy(d_means, h_means, n_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sds, h_sds, n_cols * sizeof(double), cudaMemcpyHostToDevice));

    // --- Apply Scaling ---
    dim3 threadsPerBlockScale(16, 16); // 2D block
    dim3 blocksPerGridScale( (n_cols + threadsPerBlockScale.x - 1) / threadsPerBlockScale.x,
                             (n_rows + threadsPerBlockScale.y - 1) / threadsPerBlockScale.y );
    applyColScalingDenseKernel<<<blocksPerGridScale, threadsPerBlockScale>>>(d_matrix, n_rows, n_cols, d_means, d_sds);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for scaling kernel

    // --- Cleanup ---
    cudaFree(d_sums);
    cudaFree(d_sums_sq);
    free(h_sums);
    free(h_sums_sq);
    free(h_means);
    free(h_sds);

    return 0; // Success
}

// Host function for sparse scaling
extern "C" int ridge_cuda_scale_sparse_matrix_csc(
    double *d_vals,
    const int *d_row_indices,
    const int *d_col_pointers,
    int n_rows, // n_rows needed for context but not directly used in scaling non-zeros
    int n_cols,
    int nnz,
    double *d_means_nz,
    double *d_sds_nz,
    int *d_counts_nz_out // Optional output
) {
     if (!cuda_initialized) return -10;
     if (!d_vals || !d_row_indices || !d_col_pointers || !d_means_nz || !d_sds_nz || n_cols <= 0 || nnz < 0) return -13;
     // Allow nnz == 0

    double *d_sums_nz = NULL;
    double *d_sums_sq_nz = NULL;
    int *d_counts_nz = NULL; // Internal counts

    // Allocate temporary memory
    CHECK_CUDA(cudaMalloc((void**)&d_sums_nz, n_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_sums_sq_nz, n_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_counts_nz, n_cols * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_sums_nz, 0, n_cols * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_sums_sq_nz, 0, n_cols * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_counts_nz, 0, n_cols * sizeof(int)));


    // --- Calculate Sums, Sums_sq, Counts of Non-Zeros ---
    if (nnz > 0) { // Only launch kernel if there are non-zeros
        dim3 blockDimSumsNz(SCALE_BLOCK_SIZE);
        dim3 gridDimSumsNz(n_cols); // One block per column
        calculateColSumsSparseNzKernel<<<gridDimSumsNz, blockDimSumsNz>>>(
            d_vals, d_col_pointers, n_cols, d_sums_nz, d_sums_sq_nz, d_counts_nz);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    } else {
        // If nnz is 0, means, sds, counts remain 0, which is correct.
    }

    // --- Calculate Mean_nz and SD_nz on Host ---
    double *h_sums_nz = (double*)malloc(n_cols * sizeof(double));
    double *h_sums_sq_nz = (double*)malloc(n_cols * sizeof(double));
    int *h_counts_nz = (int*)malloc(n_cols * sizeof(int));
    if (!h_sums_nz || !h_sums_sq_nz || !h_counts_nz) {
        cudaFree(d_sums_nz); cudaFree(d_sums_sq_nz); cudaFree(d_counts_nz);
        free(h_sums_nz); free(h_sums_sq_nz); free(h_counts_nz);
        fprintf(stderr, "Host allocation failed in scale_sparse\n"); return -1;
    }
    CHECK_CUDA(cudaMemcpy(h_sums_nz, d_sums_nz, n_cols * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sums_sq_nz, d_sums_sq_nz, n_cols * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_counts_nz, d_counts_nz, n_cols * sizeof(int), cudaMemcpyDeviceToHost));

    double *h_means_nz = (double*)malloc(n_cols * sizeof(double));
    double *h_sds_nz = (double*)malloc(n_cols * sizeof(double));
     if (!h_means_nz || !h_sds_nz) {
         cudaFree(d_sums_nz); cudaFree(d_sums_sq_nz); cudaFree(d_counts_nz);
         free(h_sums_nz); free(h_sums_sq_nz); free(h_counts_nz); free(h_means_nz); free(h_sds_nz);
         fprintf(stderr, "Host allocation failed for mean/sd in scale_sparse\n"); return -1;
     }

    for (int j = 0; j < n_cols; ++j) {
        int count = h_counts_nz[j];
        if (count > 0) {
            double count_d = (double)count;
            h_means_nz[j] = h_sums_nz[j] / count_d;
            double variance_nz = (h_sums_sq_nz[j] / count_d) - (h_means_nz[j] * h_means_nz[j]);

            if (variance_nz < 0.0 && variance_nz > -EPS) variance_nz = 0.0;
            if (variance_nz < 0.0) {
                 fprintf(stderr, "Warning: Negative variance_nz (%.4e) encountered in sparse scaling col %d (count %d). Clamping to 0.\n", variance_nz, j, count);
                 variance_nz = 0.0;
            }

            // Use population SD based on non-zeros
            h_sds_nz[j] = sqrt(variance_nz);
            // Correct SD if count is 1 (SD should be 0)
            if (count == 1) {
                 h_sds_nz[j] = 0.0;
                 // Mean is just the single value, variance is 0.
            }
        } else {
            // No non-zeros in this column
            h_means_nz[j] = 0.0;
            h_sds_nz[j] = 0.0;
        }
    }

    // Copy calculated means and SDs to the output device pointers
    CHECK_CUDA(cudaMemcpy(d_means_nz, h_means_nz, n_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sds_nz, h_sds_nz, n_cols * sizeof(double), cudaMemcpyHostToDevice));

    // Optionally copy counts back if requested
    if (d_counts_nz_out != NULL) {
        CHECK_CUDA(cudaMemcpy(d_counts_nz_out, d_counts_nz, n_cols * sizeof(int), cudaMemcpyDeviceToDevice));
    }

    // --- Apply Sparse Scaling ---
    if (nnz > 0) {
        dim3 blockDimScaleNz(SCALE_BLOCK_SIZE);
        dim3 gridDimScaleNz(n_cols); // One block per column
        applyColScalingSparseNzKernel<<<gridDimScaleNz, blockDimScaleNz>>>(
            d_vals, d_col_pointers, n_cols, d_means_nz, d_sds_nz);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // --- Cleanup ---
    cudaFree(d_sums_nz);
    cudaFree(d_sums_sq_nz);
    cudaFree(d_counts_nz); // Free the internal counts buffer
    free(h_sums_nz);
    free(h_sums_sq_nz);
    free(h_counts_nz);
    free(h_means_nz);
    free(h_sds_nz);

    return 0; // Success
}

//----------------------------------------------------------------------------//
// RIDGE REGRESSION IMPLEMENTATIONS                                           //
//----------------------------------------------------------------------------//

/**
 * CUDA implementation of ridge regression with dense matrices (Permutation Test only).
 * Assumes inputs X, Y and output beta are COLUMN-MAJOR.
 * Implements batch processing to reduce memory usage for large datasets.
 */
int ridge_cuda_dense(
    const double *X, /* n_genes x n_features */
    const double *Y, /* n_genes x n_samples */
    int n_genes, int n_features, int n_samples,
    double lambda_val, int n_rand, // n_rand must be > 0
    int batch_size,   // New parameter: columns per batch, 0 means process all at once
    double *beta,   /* n_features x n_samples */
    double *se,     /* n_features x n_samples */
    double *zscore, /* n_features x n_samples */
    double *pvalue /* n_features x n_samples */
) {
    // --- Input Validation ---
    if (!cuda_initialized) { fprintf(stderr, "Error: CUDA not initialized. Call ridge_cuda_init() first.\n"); return -10; }
    if (n_genes <= 0 || n_features <= 0 || n_samples <= 0) { fprintf(stderr, "Error: Input dimensions must be positive (n_genes=%d, n_features=%d, n_samples=%d).\n", n_genes, n_features, n_samples); return -11; }
    if (lambda_val < 0.0) { fprintf(stderr, "Error: Lambda must be non-negative (lambda=%.4e).\n", lambda_val); return -12; }
    if (X == NULL || Y == NULL || beta == NULL || se == NULL || zscore == NULL || pvalue == NULL) { fprintf(stderr, "Error: NULL pointer provided for required input/output arrays.\n"); return -13; }
    if (n_rand <= 0) { // Permutation test ONLY
        fprintf(stderr, "Error: n_rand must be positive for permutation testing in ridge_cuda_dense (n_rand=%d).\n", n_rand);
        return -14; // Invalid n_rand
    }
    
    // --- Process batch_size parameter ---
    if (batch_size <= 0 || batch_size > n_samples) {
        // Use all samples in a single batch if batch_size is invalid
        batch_size = n_samples;
    }
    int n_batches = (n_samples + batch_size - 1) / batch_size; // Ceiling division
    printf("Processing %d samples in %d batches of up to %d samples each\n", 
           n_samples, n_batches, batch_size);

    // --- Variable Declaration ---
    double *d_X = NULL, *d_X_transpose = NULL, *d_XtX = NULL;
    double *d_T = NULL, *d_workspace = NULL;
    int *d_info = NULL;
    
    // Batch-specific variables
    double *d_Y_batch = NULL, *d_beta_batch = NULL;
    double *d_T_permuted = NULL, *d_beta_perm = NULL, *d_abs_beta_obs = NULL;
    double *d_sum_b = NULL, *d_sum_b2 = NULL, *d_count_ge = NULL;
    int *d_indices = NULL;
    int *h_indices = NULL; // Host permutation indices
    
    // Per-batch accumulation buffers
    double *h_sum_b = NULL, *h_sum_b2 = NULL, *h_count_ge = NULL;
    double n_rand_d = (double)n_rand; // Move these declarations up before any goto
    double n_rand_plus_1 = (double)(n_rand + 1);
    
    int exit_status = 0;
    double alpha = 1.0, beta_zero = 0.0;
    int compute_block_size = 256;
    int abs_num_blocks; // Declare here for reuse

    // Define dimensions for BLAS/LAPACK (Column-Major)
    int p = n_features; // Number of features/predictors
    int n = n_genes;    // Number of observations/genes
    int m = n_samples;  // Number of response variables/samples
    int m_batch = batch_size; // Max samples per batch

    cudaStream_t stream = NULL;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));
    CHECK_CUSOLVER(cusolverDnSetStream(cusolver_handle, stream));

    // --- Memory Allocation for X and common data structures ---
    CHECK_CUDA(cudaMalloc((void**)&d_X, (size_t)n * p * sizeof(double)));          // n x p
    CHECK_CUDA(cudaMalloc((void**)&d_X_transpose, (size_t)p * n * sizeof(double))); // p x n
    CHECK_CUDA(cudaMalloc((void**)&d_XtX, (size_t)p * p * sizeof(double)));        // p x p
    CHECK_CUDA(cudaMalloc((void**)&d_T, (size_t)p * n * sizeof(double)));          // p x n
    CHECK_CUDA(cudaMalloc((void**)&d_info, sizeof(int)));
    
    // Allocate permutation indices once
    CHECK_CUDA(cudaMalloc((void**)&d_indices, n * sizeof(int)));                   // size n (permute rows of X or columns of T)
    
    // --- Batch-specific allocations ---
    CHECK_CUDA(cudaMalloc((void**)&d_Y_batch, (size_t)n * batch_size * sizeof(double)));          // n x batch_size
    CHECK_CUDA(cudaMalloc((void**)&d_beta_batch, (size_t)p * batch_size * sizeof(double))); // p x batch_size
    CHECK_CUDA(cudaMalloc((void**)&d_T_permuted, (size_t)p * n * sizeof(double))); // p x n
    CHECK_CUDA(cudaMalloc((void**)&d_beta_perm, (size_t)p * batch_size * sizeof(double))); // p x batch_size
    CHECK_CUDA(cudaMalloc((void**)&d_abs_beta_obs, (size_t)p * batch_size * sizeof(double))); // p x batch_size
    CHECK_CUDA(cudaMalloc((void**)&d_sum_b, (size_t)p * batch_size * sizeof(double))); // p x batch_size
    CHECK_CUDA(cudaMalloc((void**)&d_sum_b2, (size_t)p * batch_size * sizeof(double)));// p x batch_size
    CHECK_CUDA(cudaMalloc((void**)&d_count_ge, (size_t)p * batch_size * sizeof(double)));// p x batch_size

    // --- Data Transfer for X ---
    CHECK_CUDA(cudaMemcpy(d_X, X, (size_t)n * p * sizeof(double), cudaMemcpyHostToDevice));

    // --- Core Ridge Calculation (X part - done once for all batches) ---
    // 1. X_transpose = X^T (result is p x n, stored column-major)
    CHECK_CUBLAS(cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, p, n, &alpha, d_X, n, &beta_zero, NULL, p, d_X_transpose, p));
    
    // 2. XtX = X_transpose * X ( (p x n) * (n x p) = (p x p) )
    CHECK_CUBLAS(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, n, &alpha, d_X_transpose, p, d_X, n, &beta_zero, d_XtX, p));
    
    // 3. Add lambda to diagonal of XtX (p x p)
    int xtx_num_blocks = (p + compute_block_size - 1) / compute_block_size;
    addDiagonalConstant<<<xtx_num_blocks, compute_block_size, 0, stream>>>(d_XtX, lambda_val, p);
    CHECK_CUDA(cudaGetLastError());
    
    // 4. Cholesky Factorization: XtX = L * L^T (or U^T * U)
    int workspace_size = 0;
    CHECK_CUSOLVER(cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_UPPER, p, d_XtX, p, &workspace_size));
    CHECK_CUDA(cudaMalloc((void**)&d_workspace, workspace_size * sizeof(double)));
    CHECK_CUSOLVER(cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_UPPER, p, d_XtX, p, d_workspace, workspace_size, d_info));
    
    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if (h_info != 0) {
        fprintf(stderr, "Error: Cholesky factorization failed (info = %d). Matrix might be singular or not positive definite.\n", h_info);
        exit_status = 10; goto cleanup;
    }
    
    // 5. Solve for T using Cholesky factor: (X^T*X + lambda*I) * T = X^T => U^T * U * T = X^T
    CHECK_CUDA(cudaMemcpy(d_T, d_X_transpose, (size_t)p * n * sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK_CUSOLVER(cusolverDnDpotrs(cusolver_handle, CUBLAS_FILL_MODE_UPPER, p, n, d_XtX, p, d_T, p, d_info));
    
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if (h_info != 0) {
        fprintf(stderr, "Error: Solving system with Cholesky factor failed (info = %d).\n", h_info);
        exit_status = 11; goto cleanup;
    }

    // --- Permutation Test Preparation ---
    // Allocate host memory for indices
    h_indices = (int*)malloc(n * sizeof(int)); // Permute along the n dimension (genes/observations)
    if (h_indices == NULL) {
        fprintf(stderr, "Error: Failed to allocate host memory for permutation indices.\n");
        exit_status = 12; goto cleanup;
    }
    
    // Host memory for accumulating batch results
    h_sum_b = (double*)calloc(p * m, sizeof(double));
    h_sum_b2 = (double*)calloc(p * m, sizeof(double));
    h_count_ge = (double*)calloc(p * m, sizeof(double));
    
    if (!h_sum_b || !h_sum_b2 || !h_count_ge) {
        fprintf(stderr, "Error: Failed to allocate host memory for permutation results.\n");
        exit_status = 13; goto cleanup;
    }

    // --- Process data in batches ---
    for (int batch = 0; batch < n_batches; batch++) {
        int start_col = batch * batch_size;
        int cols_in_batch = (batch < n_batches - 1 || m % batch_size == 0) ? 
                            batch_size : m % batch_size;
        
        printf("Processing batch %d/%d: columns %d to %d\n", 
               batch+1, n_batches, start_col+1, start_col+cols_in_batch);
        
        // --- Batch Data Transfer ---
        CHECK_CUDA(cudaMemcpy(d_Y_batch, Y + (size_t)start_col * n, 
                             (size_t)n * cols_in_batch * sizeof(double), 
                             cudaMemcpyHostToDevice));
        
        // --- Compute beta for current batch ---
        // 6. Compute beta_batch = T * Y_batch ( (p x n) * (n x cols_in_batch) = (p x cols_in_batch) )
        CHECK_CUBLAS(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, p, cols_in_batch, n, 
                                &alpha, d_T, p, d_Y_batch, n, &beta_zero, d_beta_batch, p));
        
        // --- Copy batch beta results ---
        CHECK_CUDA(cudaMemcpy(beta + (size_t)start_col * p, d_beta_batch, 
                              (size_t)p * cols_in_batch * sizeof(double), 
                              cudaMemcpyDeviceToHost));
        
        // --- Permutation Test for current batch ---
        // Calculate absolute values of observed beta
        abs_num_blocks = ((size_t)p * cols_in_batch + compute_block_size - 1) / compute_block_size;
        calculateAbsValues<<<abs_num_blocks, compute_block_size, 0, stream>>>(
            d_beta_batch, d_abs_beta_obs, (size_t)p * cols_in_batch);
        CHECK_CUDA(cudaGetLastError());
        
        // Initialize permutation statistics accumulators for this batch
        CHECK_CUDA(cudaMemset(d_sum_b, 0, (size_t)p * cols_in_batch * sizeof(double)));
        CHECK_CUDA(cudaMemset(d_sum_b2, 0, (size_t)p * cols_in_batch * sizeof(double)));
        CHECK_CUDA(cudaMemset(d_count_ge, 0, (size_t)p * cols_in_batch * sizeof(double)));
        
        // Permutation loop
        for (int r = 0; r < n_rand; r++) {
            // Generate permutation indices on host
            for (int i = 0; i < n; i++) h_indices[i] = i;
            fisher_yates_shuffle(h_indices, n);
            CHECK_CUDA(cudaMemcpy(d_indices, h_indices, n * sizeof(int), cudaMemcpyHostToDevice));
            
            // Permute columns of T (p x n) based on row indices of original X/Y
            permute_T_columns_cuda(d_T_permuted, d_T, d_indices, p, n);
            CHECK_CUDA(cudaGetLastError());
            
            // Recalculate beta_perm = T_permuted * Y_batch
            CHECK_CUBLAS(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, p, cols_in_batch, n, 
                                    &alpha, d_T_permuted, p, d_Y_batch, n, &beta_zero, d_beta_perm, p));
            
            // Update permutation statistics on GPU
            int stats_num_blocks = ((size_t)p * cols_in_batch + compute_block_size - 1) / compute_block_size;
            updatePermutationStats<<<stats_num_blocks, compute_block_size, 0, stream>>>(
                d_beta_perm, d_abs_beta_obs, d_sum_b, d_sum_b2, d_count_ge, 
                (size_t)p * cols_in_batch, EPS);
            CHECK_CUDA(cudaGetLastError());
        }
        
        // --- Copy batch permutation stats to host ---
        double *batch_sum_b = (double*)malloc((size_t)p * cols_in_batch * sizeof(double));
        double *batch_sum_b2 = (double*)malloc((size_t)p * cols_in_batch * sizeof(double));
        double *batch_count_ge = (double*)malloc((size_t)p * cols_in_batch * sizeof(double));
        
        if (!batch_sum_b || !batch_sum_b2 || !batch_count_ge) {
            fprintf(stderr, "Error: Failed to allocate host memory for batch permutation results.\n");
            free(batch_sum_b); free(batch_sum_b2); free(batch_count_ge);
            exit_status = 13; goto cleanup;
        }
        
        CHECK_CUDA(cudaMemcpy(batch_sum_b, d_sum_b, (size_t)p * cols_in_batch * sizeof(double), 
                             cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(batch_sum_b2, d_sum_b2, (size_t)p * cols_in_batch * sizeof(double), 
                             cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(batch_count_ge, d_count_ge, (size_t)p * cols_in_batch * sizeof(double), 
                             cudaMemcpyDeviceToHost));
        
        // --- Accumulate batch results in the full arrays ---
        for (int j = 0; j < cols_in_batch; j++) {
            for (int i = 0; i < p; i++) {
                size_t src_idx = (size_t)j * p + i;
                size_t dst_idx = (size_t)(start_col + j) * p + i;
                h_sum_b[dst_idx] = batch_sum_b[src_idx];
                h_sum_b2[dst_idx] = batch_sum_b2[src_idx];
                h_count_ge[dst_idx] = batch_count_ge[src_idx];
            }
        }
        
        free(batch_sum_b); free(batch_sum_b2); free(batch_count_ge);
    }
    
    // --- Calculate Final Statistics ---
    for (size_t idx = 0; idx < (size_t)p * m; idx++) {
        double mean_perm = h_sum_b[idx] / n_rand_d;
        double mean_sq_perm = h_sum_b2[idx] / n_rand_d;
        // Variance calculation: Var = E[X^2] - (E[X])^2
        double var_perm = mean_sq_perm - (mean_perm * mean_perm);
        // Handle potential small negative variance due to floating point errors
        if (var_perm < 0.0) {
             if (var_perm < -EPS * 100) { // Only warn if significantly negative
                 fprintf(stderr, "Warning: Negative variance (%e) calculated at index %zu. Clamping to 0.\n", var_perm, idx);
             }
             var_perm = 0.0;
        }
        se[idx] = sqrt(var_perm);

        // Calculate Z-score
        double se_val = se[idx];
        double beta_obs = beta[idx];
        // Use relative tolerance for division by zero check
        double rel_eps = fmax(EPS, fabs(beta_obs - mean_perm) * EPS * 10.0);
        if (se_val > rel_eps) {
            zscore[idx] = (beta_obs - mean_perm) / se_val;
        } else {
             // If SE is near zero, check if mean is also near zero
             if (fabs(beta_obs - mean_perm) < rel_eps) {
                 zscore[idx] = 0.0; // Beta is effectively same as permuted mean
             } else {
                 zscore[idx] = (beta_obs > mean_perm) ? INFINITY : -INFINITY; // Beta differs significantly from mean, SE is zero -> infinite Z
             }
        }

        // Calculate P-value: (count(|beta_perm| >= |beta_obs|) + 1) / (n_rand + 1)
        double p_raw = (h_count_ge[idx] + 1.0) / n_rand_plus_1;
        pvalue[idx] = fmin(1.0, fmax(0.0, p_raw)); // Clamp between 0 and 1
    }

cleanup:
    // --- Cleanup ---
    if (stream) cudaStreamDestroy(stream);
    cudaFree(d_X); cudaFree(d_X_transpose); cudaFree(d_XtX);
    cudaFree(d_T); cudaFree(d_workspace); cudaFree(d_info);
    cudaFree(d_Y_batch); cudaFree(d_beta_batch);
    cudaFree(d_T_permuted); cudaFree(d_beta_perm); cudaFree(d_indices);
    cudaFree(d_abs_beta_obs); cudaFree(d_sum_b); cudaFree(d_sum_b2); cudaFree(d_count_ge);
    
    // Free host memory
    if (h_indices != NULL) free(h_indices);
    if (h_sum_b != NULL) free(h_sum_b);
    if (h_sum_b2 != NULL) free(h_sum_b2);
    if (h_count_ge != NULL) free(h_count_ge);

    return exit_status;
}


/**
 * CUDA implementation of ridge regression with sparse Y (CSC format) (Permutation Test only).
 * Assumes input X and output beta are COLUMN-MAJOR.
 * Assumes input Y (CSC) indices are 0-based.
 * Implements batch processing to reduce memory usage for large datasets.
 */
int ridge_cuda_sparse(
    const double *X, /* n_genes x n_features */
    int n_genes, int n_features,
    const double *Y_vals, const int *Y_row_indices, const int *Y_col_pointers,
    int n_samples, int nnz,
    double lambda_val, int n_rand, // n_rand must be > 0
    int batch_size,   // New parameter: columns per batch, 0 means process all at once
    double *beta,   /* n_features x n_samples */
    double *se,     /* n_features x n_samples */
    double *zscore, /* n_features x n_samples */
    double *pvalue  /* n_features x n_samples */
) {
    // --- Input Validation ---
    if (!cuda_initialized) { fprintf(stderr, "Error: CUDA not initialized. Call ridge_cuda_init() first.\n"); return -10; }
    if (n_genes <= 0 || n_features <= 0 || n_samples <= 0 || nnz <= 0) { fprintf(stderr, "Error: Input dimensions must be positive (n_genes=%d, n_features=%d, n_samples=%d, nnz=%d).\n", n_genes, n_features, n_samples, nnz); return -11; }
    if (lambda_val < 0.0) { fprintf(stderr, "Error: Lambda must be non-negative (lambda=%.4e).\n", lambda_val); return -12; }
    if (Y_vals == NULL || Y_row_indices == NULL || Y_col_pointers == NULL || X == NULL || beta == NULL || se == NULL || zscore == NULL || pvalue == NULL) { fprintf(stderr, "Error: NULL pointer provided for required input/output arrays.\n"); return -13; }
    if (n_rand <= 0) { // Permutation test ONLY
        fprintf(stderr, "Error: n_rand must be positive for permutation testing in ridge_cuda_sparse (n_rand=%d).\n", n_rand);
        return -14;
    }
    
    // --- Process batch_size parameter ---
    if (batch_size <= 0 || batch_size > n_samples) {
        // Use all samples in a single batch if batch_size is invalid
        batch_size = n_samples;
    }
    int n_batches = (n_samples + batch_size - 1) / batch_size; // Ceiling division
    printf("Processing %d samples in %d batches of up to %d samples each\n", 
           n_samples, n_batches, batch_size);

    // --- Variable Declaration ---
    double *d_X = NULL, *d_XtX = NULL, *d_X_transpose = NULL, *d_T = NULL;
    double *d_beta = NULL, *d_workspace = NULL;
    int *d_info = NULL;
    
    // Sparse Y structures
    double *d_Y_vals = NULL; 
    int *d_Y_row_indices = NULL, *d_Y_col_pointers = NULL;
    
    // Batch-specific structures
    cusparseSpMatDescr_t Y_sparse_csc_batch = NULL;
    cusparseDnMatDescr_t T_transpose_dense = NULL, beta_transpose_dense = NULL;
    double *d_T_transpose = NULL, *d_beta_transpose = NULL;
    double *d_T_transpose_perm = NULL, *d_beta_perm = NULL, *d_beta_perm_transpose = NULL;
    double *d_abs_beta_obs = NULL, *d_sum_b = NULL, *d_sum_b2 = NULL, *d_count_ge = NULL;
    int *d_indices = NULL;
    void *d_cusparse_buffer = NULL; size_t cusparse_buffer_size = 0;
    
    // Host buffers for accumulation
    int *h_indices = NULL; // Host permutation indices
    double *h_sum_b = NULL, *h_sum_b2 = NULL, *h_count_ge = NULL; // Host results
    
    // Descriptors needed inside loop (declare outside)
    cusparseDnMatDescr_t beta_perm_transpose_dense = NULL;
    cusparseDnMatDescr_t T_transpose_perm_dense = NULL;

    // Move these early to avoid goto initialization bypass errors
    double n_rand_d = (double)n_rand;
    double n_rand_plus_1 = (double)(n_rand + 1);

    size_t total_beta_elements = (size_t)n_features * n_samples;
    int exit_status = 0; double alpha = 1.0, beta_zero = 0.0;
    int compute_block_size = 256;

    // Define dimensions for BLAS/LAPACK/SPARSE (Column-Major base)
    int p = n_features; // Number of features/predictors
    int n = n_genes;    // Number of observations/genes
    int m = n_samples;  // Number of response variables/samples
    int m_batch = batch_size; // Max samples per batch

    // --- Initialization (Stream for Async Operations) ---
    cudaStream_t stream = NULL;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));
    CHECK_CUSOLVER(cusolverDnSetStream(cusolver_handle, stream));
    CHECK_CUSPARSE(cusparseSetStream(cusparse_handle, stream));

    // --- Memory Allocation (Common parts) ---
    CHECK_CUDA(cudaMallocAsync((void**)&d_X, (size_t)n * p * sizeof(double), stream));          // n x p
    CHECK_CUDA(cudaMallocAsync((void**)&d_XtX, (size_t)p * p * sizeof(double), stream));        // p x p
    CHECK_CUDA(cudaMallocAsync((void**)&d_X_transpose, (size_t)p * n * sizeof(double), stream)); // p x n
    CHECK_CUDA(cudaMallocAsync((void**)&d_T, (size_t)p * n * sizeof(double), stream));          // p x n
    CHECK_CUDA(cudaMallocAsync((void**)&d_info, sizeof(int), stream));
    
    // Entire sparse Y structure (all columns)
    CHECK_CUDA(cudaMallocAsync((void**)&d_Y_vals, nnz * sizeof(double), stream));
    CHECK_CUDA(cudaMallocAsync((void**)&d_Y_row_indices, nnz * sizeof(int), stream));
    CHECK_CUDA(cudaMallocAsync((void**)&d_Y_col_pointers, (m + 1) * sizeof(int), stream));
    
    // Intermediates for SpMM
    CHECK_CUDA(cudaMallocAsync((void**)&d_T_transpose, (size_t)n * p * sizeof(double), stream)); // n x p
    // Only allocate for batch size
    CHECK_CUDA(cudaMallocAsync((void**)&d_beta_transpose, (size_t)batch_size * p * sizeof(double), stream)); // batch_size x p
    
    // Permutation (Async)
    CHECK_CUDA(cudaMallocAsync((void**)&d_T_transpose_perm, (size_t)n * p * sizeof(double), stream)); // n x p
    // Only allocate for batch size
    CHECK_CUDA(cudaMallocAsync((void**)&d_beta_perm, (size_t)p * batch_size * sizeof(double), stream)); // p x batch_size
    CHECK_CUDA(cudaMallocAsync((void**)&d_beta_perm_transpose, (size_t)batch_size * p * sizeof(double), stream)); // batch_size x p
    
    CHECK_CUDA(cudaMallocAsync((void**)&d_indices, n * sizeof(int), stream));                   // size n
    // Only allocate for batch size
    CHECK_CUDA(cudaMallocAsync((void**)&d_abs_beta_obs, (size_t)p * batch_size * sizeof(double), stream)); // p x batch_size
    CHECK_CUDA(cudaMallocAsync((void**)&d_sum_b, (size_t)p * batch_size * sizeof(double), stream)); // p x batch_size
    CHECK_CUDA(cudaMallocAsync((void**)&d_sum_b2, (size_t)p * batch_size * sizeof(double), stream));// p x batch_size
    CHECK_CUDA(cudaMallocAsync((void**)&d_count_ge, (size_t)p * batch_size * sizeof(double), stream));// p x batch_size

    // --- Data Transfer (Async) ---
    CHECK_CUDA(cudaMemcpyAsync(d_X, X, (size_t)n * p * sizeof(double), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_Y_vals, Y_vals, nnz * sizeof(double), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_Y_row_indices, Y_row_indices, nnz * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_Y_col_pointers, Y_col_pointers, (m + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));

    // --- Create cuSPARSE Descriptors for entire sparse Y ---
    // Y_sparse_csc: n rows, m cols, CSC format
    CHECK_CUSPARSE(cusparseCreateCsc(&Y_sparse_csc_batch, n, m, nnz, d_Y_col_pointers, d_Y_row_indices, d_Y_vals, 
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    
    // T_transpose is (n x p), lda=n (column major)
    CHECK_CUSPARSE(cusparseCreateDnMat(&T_transpose_dense, n, p, n, d_T_transpose, CUDA_R_64F, CUSPARSE_ORDER_COL));
    
    // beta_transpose_dense: batch_size rows, p cols, lda=batch_size (column major)
    CHECK_CUSPARSE(cusparseCreateDnMat(&beta_transpose_dense, batch_size, p, batch_size, d_beta_transpose, CUDA_R_64F, CUSPARSE_ORDER_COL));
    
    // Create descriptors needed inside loop here to avoid re-creation
    // T_transpose_perm_dense: n rows, p cols, lda=n (column major)
    CHECK_CUSPARSE(cusparseCreateDnMat(&T_transpose_perm_dense, n, p, n, d_T_transpose_perm, CUDA_R_64F, CUSPARSE_ORDER_COL));
    
    // beta_perm_transpose_dense: batch_size rows, p cols, lda=batch_size (column major)
    CHECK_CUSPARSE(cusparseCreateDnMat(&beta_perm_transpose_dense, batch_size, p, batch_size, d_beta_perm_transpose, CUDA_R_64F, CUSPARSE_ORDER_COL));

    // --- Core Ridge Calculation (X part - Steps 1-5: Compute T, Async) ---
    // 1. X_transpose = X^T (p x n)
    CHECK_CUBLAS(cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, p, n, &alpha, d_X, n, &beta_zero, NULL, p, d_X_transpose, p));
    
    // 2. XtX = X_transpose * X (p x p)
    CHECK_CUBLAS(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, p, p, n, &alpha, d_X_transpose, p, d_X, n, &beta_zero, d_XtX, p));
    
    // 3. Add lambda to diagonal of XtX (p x p)
    int xtx_num_blocks = (p + compute_block_size - 1) / compute_block_size;
    addDiagonalConstant<<<xtx_num_blocks, compute_block_size, 0, stream>>>(d_XtX, lambda_val, p);
    CHECK_CUDA(cudaGetLastError());
    
    // 4. Cholesky Factorization: XtX = U^T * U
    int workspace_size = 0;
    CHECK_CUSOLVER(cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_UPPER, p, d_XtX, p, &workspace_size));
    CHECK_CUDA(cudaMallocAsync((void**)&d_workspace, workspace_size * sizeof(double), stream));
    CHECK_CUSOLVER(cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_UPPER, p, d_XtX, p, d_workspace, workspace_size, d_info));
    
    // Need to synchronize before checking h_info
    int h_info = 0;
    CHECK_CUDA(cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream)); // Wait for potrf and memcpy D->H to complete
    if (h_info != 0) {
        fprintf(stderr, "Error: Cholesky factorization failed (info = %d). Matrix might be singular or not positive definite.\n", h_info);
        exit_status = 10; goto cleanup;
    }
    
    // 5. Solve for T using Cholesky factor: U^T * U * T = X^T
    CHECK_CUDA(cudaMemcpyAsync(d_T, d_X_transpose, (size_t)p * n * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    CHECK_CUSOLVER(cusolverDnDpotrs(cusolver_handle, CUBLAS_FILL_MODE_UPPER, p, n, d_XtX, p, d_T, p, d_info));
    
    // Need to synchronize before checking h_info again
    CHECK_CUDA(cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream)); // Wait for potrs and memcpy D->H to complete
    if (h_info != 0) {
        fprintf(stderr, "Error: Solving system with Cholesky factor failed (info = %d).\n", h_info);
        exit_status = 11; goto cleanup;
    }
    
    // 6a. Compute T_transpose = T^T (n x p)
    // Input T is (p x n), lda=p. Output T_transpose is (n x p), ldc=n.
    CHECK_CUBLAS(cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, p, &alpha, d_T, p, &beta_zero, NULL, n, d_T_transpose, n));
    
    // --- Set up for SpMM buffer allocation ---
    // We only need to do this once since we're using the same dimensions for all batches
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, Y_sparse_csc_batch, T_transpose_dense, &beta_zero, beta_transpose_dense,
                                           CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &cusparse_buffer_size));
    CHECK_CUDA(cudaMallocAsync(&d_cusparse_buffer, cusparse_buffer_size, stream));
    
    // --- Prepare for permutation test ---
    // Allocate host memory for indices
    h_indices = (int*)malloc(n * sizeof(int)); // Permute along the n dimension (genes/observations)
    if (h_indices == NULL) {
        fprintf(stderr, "Error: Failed to allocate host memory for permutation indices.\n");
        exit_status = 12; goto cleanup;
    }
    
    // Allocate host buffers for accumulating permutation results
    h_sum_b = (double*)calloc(total_beta_elements, sizeof(double));
    h_sum_b2 = (double*)calloc(total_beta_elements, sizeof(double));
    h_count_ge = (double*)calloc(total_beta_elements, sizeof(double));
    
    if (!h_sum_b || !h_sum_b2 || !h_count_ge) {
        fprintf(stderr, "Error: Failed to allocate host memory for permutation results.\n");
        exit_status = 13; goto cleanup;
    }
    
    // --- Process each batch of Y columns ---
    for (int batch = 0; batch < n_batches; batch++) {
        int start_col = batch * batch_size;
        int cols_in_batch = (batch < n_batches - 1 || m % batch_size == 0) ? 
                            batch_size : m % batch_size;
        
        printf("Processing batch %d/%d: columns %d to %d\n", 
               batch+1, n_batches, start_col+1, start_col+cols_in_batch);
        
        // --- Instead of using cusparseDnMatSetDimensions, recreate the matrices ---
        // The current version of CUDA doesn't support this function or it has a different name
        
        // Destroy previous descriptors if they exist (except for first batch)
        if (batch > 0) {
            if (beta_transpose_dense) cusparseDestroyDnMat(beta_transpose_dense);
            if (beta_perm_transpose_dense) cusparseDestroyDnMat(beta_perm_transpose_dense);
        }
        
        // Recreate with new dimensions
        CHECK_CUSPARSE(cusparseCreateDnMat(&beta_transpose_dense, cols_in_batch, p, cols_in_batch, 
                                          d_beta_transpose, CUDA_R_64F, CUSPARSE_ORDER_COL));
        CHECK_CUSPARSE(cusparseCreateDnMat(&beta_perm_transpose_dense, cols_in_batch, p, cols_in_batch, 
                                          d_beta_perm_transpose, CUDA_R_64F, CUSPARSE_ORDER_COL));
        
        // --- Compute beta for current batch using SpMM ---
        // We need to extract the submatrix of Y for this batch
        // Create a view of the Y sparse matrix for just this batch of columns
        cusparseSpMatDescr_t Y_sparse_csc_batch_view = NULL;
        CHECK_CUSPARSE(cusparseCreateCsc(&Y_sparse_csc_batch_view, n, cols_in_batch, 
                                        Y_col_pointers[start_col + cols_in_batch] - Y_col_pointers[start_col],
                                        d_Y_col_pointers + start_col, 
                                        d_Y_row_indices + Y_col_pointers[start_col],
                                        d_Y_vals + Y_col_pointers[start_col],
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, 
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        
        // 6b. Compute beta_transpose_batch = Y_batch^T * T_transpose ( (cols_in_batch x n) * (n x p) = (cols_in_batch x p) )
        CHECK_CUSPARSE(cusparseSpMM(cusparse_handle, 
                                    CUSPARSE_OPERATION_TRANSPOSE, // Transpose Y
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // Non-transpose T_transpose
                                    &alpha, 
                                    Y_sparse_csc_batch_view, // Batch view of sparse Y 
                                    T_transpose_dense, 
                                    &beta_zero, 
                                    beta_transpose_dense,
                                    CUDA_R_64F, 
                                    CUSPARSE_SPMM_ALG_DEFAULT, 
                                    d_cusparse_buffer));
        
        // Create a temporary buffer for the batch beta
        double *d_beta_batch = NULL;
        CHECK_CUDA(cudaMallocAsync((void**)&d_beta_batch, (size_t)p * cols_in_batch * sizeof(double), stream));
        
        // 6c. Compute beta_batch = (beta_transpose_batch)^T ( (cols_in_batch x p)^T = (p x cols_in_batch) )
        // Input beta_transpose is (cols_in_batch x p), lda=cols_in_batch. Output beta_batch is (p x cols_in_batch), ldc=p.
        CHECK_CUBLAS(cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, p, cols_in_batch, &alpha, d_beta_transpose, cols_in_batch, &beta_zero, NULL, p, d_beta_batch, p));
        
        // --- Copy batch beta to host ---
        CHECK_CUDA(cudaMemcpyAsync(beta + (size_t)start_col * p, d_beta_batch, 
                                  (size_t)p * cols_in_batch * sizeof(double), 
                                  cudaMemcpyDeviceToHost, stream));
        
        // --- Permutation Test for current batch ---
        // Calculate absolute values of observed beta
        int abs_num_blocks = ((size_t)p * cols_in_batch + compute_block_size - 1) / compute_block_size;
        calculateAbsValues<<<abs_num_blocks, compute_block_size, 0, stream>>>(
            d_beta_batch, d_abs_beta_obs, (size_t)p * cols_in_batch);
        CHECK_CUDA(cudaGetLastError());
        
        // Initialize permutation statistics accumulators for this batch
        CHECK_CUDA(cudaMemsetAsync(d_sum_b, 0, (size_t)p * cols_in_batch * sizeof(double), stream));
        CHECK_CUDA(cudaMemsetAsync(d_sum_b2, 0, (size_t)p * cols_in_batch * sizeof(double), stream));
        CHECK_CUDA(cudaMemsetAsync(d_count_ge, 0, (size_t)p * cols_in_batch * sizeof(double), stream));
        
        // Permutation loop
        for (int r = 0; r < n_rand; r++) {
            // Generate permutation indices on host
            for (int i = 0; i < n; i++) h_indices[i] = i;
            fisher_yates_shuffle(h_indices, n);
            CHECK_CUDA(cudaMemcpyAsync(d_indices, h_indices, n * sizeof(int), cudaMemcpyHostToDevice, stream));
            
            // Permute rows of T_transpose (n x p)
            permute_T_transpose_rows_cuda(d_T_transpose_perm, d_T_transpose, d_indices, n, p);
            CHECK_CUDA(cudaGetLastError());
            
            // Recalculate beta_perm = (Y_batch^T * T_transpose_perm)^T
            CHECK_CUSPARSE(cusparseSpMM(cusparse_handle, 
                                       CUSPARSE_OPERATION_TRANSPOSE, 
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, 
                                       Y_sparse_csc_batch_view, 
                                       T_transpose_perm_dense, 
                                       &beta_zero, 
                                       beta_perm_transpose_dense,
                                       CUDA_R_64F, 
                                       CUSPARSE_SPMM_ALG_DEFAULT, 
                                       d_cusparse_buffer));
            
            // Compute beta_perm = (beta_perm_transpose)^T ( (cols_in_batch x p)^T = (p x cols_in_batch) )
            CHECK_CUBLAS(cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, p, cols_in_batch, &alpha, d_beta_perm_transpose, cols_in_batch, &beta_zero, NULL, p, d_beta_perm, p));
            
            // Update permutation statistics on GPU
            int stats_num_blocks = ((size_t)p * cols_in_batch + compute_block_size - 1) / compute_block_size;
            updatePermutationStats<<<stats_num_blocks, compute_block_size, 0, stream>>>(
                d_beta_perm, d_abs_beta_obs, d_sum_b, d_sum_b2, d_count_ge, 
                (size_t)p * cols_in_batch, EPS);
            CHECK_CUDA(cudaGetLastError());
        }
        
        // --- Copy batch permutation stats to host ---
        double *batch_sum_b = (double*)malloc((size_t)p * cols_in_batch * sizeof(double));
        double *batch_sum_b2 = (double*)malloc((size_t)p * cols_in_batch * sizeof(double));
        double *batch_count_ge = (double*)malloc((size_t)p * cols_in_batch * sizeof(double));
        
        if (!batch_sum_b || !batch_sum_b2 || !batch_count_ge) {
            fprintf(stderr, "Error: Failed to allocate host memory for batch permutation results.\n");
            free(batch_sum_b); free(batch_sum_b2); free(batch_count_ge);
            cusparseDestroySpMat(Y_sparse_csc_batch_view);
            cudaFree(d_beta_batch);
            exit_status = 13; goto cleanup;
        }
        
        CHECK_CUDA(cudaMemcpyAsync(batch_sum_b, d_sum_b, (size_t)p * cols_in_batch * sizeof(double), 
                                  cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(batch_sum_b2, d_sum_b2, (size_t)p * cols_in_batch * sizeof(double), 
                                  cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(batch_count_ge, d_count_ge, (size_t)p * cols_in_batch * sizeof(double), 
                                  cudaMemcpyDeviceToHost, stream));
        
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // --- Accumulate batch results in the full arrays ---
        for (int j = 0; j < cols_in_batch; j++) {
            for (int i = 0; i < p; i++) {
                size_t src_idx = (size_t)j * p + i;
                size_t dst_idx = (size_t)(start_col + j) * p + i;
                h_sum_b[dst_idx] = batch_sum_b[src_idx];
                h_sum_b2[dst_idx] = batch_sum_b2[src_idx];
                h_count_ge[dst_idx] = batch_count_ge[src_idx];
            }
        }
        
        // Clean up batch-specific resources
        free(batch_sum_b); free(batch_sum_b2); free(batch_count_ge);
        cudaFree(d_beta_batch);
        cusparseDestroySpMat(Y_sparse_csc_batch_view);
    }
    
    // --- Calculate Final Statistics ---
    for (size_t idx = 0; idx < total_beta_elements; idx++) {
        double mean_perm = h_sum_b[idx] / n_rand_d;
        double mean_sq_perm = h_sum_b2[idx] / n_rand_d;
        double var_perm = mean_sq_perm - (mean_perm * mean_perm);
         if (var_perm < 0.0) {
             if (var_perm < -EPS * 100) { fprintf(stderr, "Warning: Negative variance (%e) calculated at index %zu. Clamping to 0.\n", var_perm, idx); }
             var_perm = 0.0;
         }
        se[idx] = sqrt(var_perm);

        double se_val = se[idx];
        double beta_obs = beta[idx];
        double rel_eps = fmax(EPS, fabs(beta_obs - mean_perm) * EPS * 10.0);
        if (se_val > rel_eps) {
            zscore[idx] = (beta_obs - mean_perm) / se_val;
        } else {
             if (fabs(beta_obs - mean_perm) < rel_eps) { zscore[idx] = 0.0; }
             else { zscore[idx] = (beta_obs > mean_perm) ? INFINITY : -INFINITY; }
        }
        double p_raw = (h_count_ge[idx] + 1.0) / n_rand_plus_1;
        pvalue[idx] = fmin(1.0, fmax(0.0, p_raw));
    }

cleanup:
    // --- Cleanup ---
    // Destroy cuSPARSE descriptors
    if (Y_sparse_csc_batch) cusparseDestroySpMat(Y_sparse_csc_batch);
    if (T_transpose_dense) cusparseDestroyDnMat(T_transpose_dense);
    if (beta_transpose_dense) cusparseDestroyDnMat(beta_transpose_dense);
    if (beta_perm_transpose_dense) cusparseDestroyDnMat(beta_perm_transpose_dense);
    if (T_transpose_perm_dense) cusparseDestroyDnMat(T_transpose_perm_dense);

    // Free CUDA memory (use cudaFree for memory allocated with cudaMallocAsync as well)
    cudaFree(d_X); cudaFree(d_XtX); cudaFree(d_X_transpose); cudaFree(d_T);
    cudaFree(d_workspace); cudaFree(d_info);
    cudaFree(d_Y_vals); cudaFree(d_Y_row_indices); cudaFree(d_Y_col_pointers);
    cudaFree(d_T_transpose); cudaFree(d_beta_transpose); cudaFree(d_cusparse_buffer);
    cudaFree(d_T_transpose_perm); cudaFree(d_beta_perm); cudaFree(d_beta_perm_transpose);
    cudaFree(d_indices); cudaFree(d_abs_beta_obs); cudaFree(d_sum_b);
    cudaFree(d_sum_b2); cudaFree(d_count_ge);

    // Free host memory
    if (h_indices != NULL) free(h_indices);
    if (h_sum_b != NULL) free(h_sum_b);
    if (h_sum_b2 != NULL) free(h_sum_b2);
    if (h_count_ge != NULL) free(h_count_ge);

    // Destroy stream
    if (stream) cudaStreamDestroy(stream);

    return exit_status;
}

} // extern "C"
