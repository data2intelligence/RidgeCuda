# RidgeRegCuda: CUDA-Accelerated Ridge Regression for R

`RidgeRegCuda` is an R package providing high-performance ridge regression using NVIDIA CUDA for GPU acceleration. The package implements ridge regression with both permutation testing and t-test significance testing, optimized for NVIDIA GPUs.

## Features

- **GPU Acceleration**: Utilizes NVIDIA CUDA for major performance improvements over CPU implementations
- **Multiple Test Methods**: Supports both permutation testing and t-test for significance testing
- **Sparse Matrix Support**: Efficient handling of sparse matrices using cuSPARSE
- **Easy R Interface**: Familiar R interface with comprehensive documentation
- **Automatic Device Selection**: Can manage multiple GPU devices

## Requirements

- R (>= 3.6.0)
- NVIDIA CUDA Toolkit (>= 10.0)
- NVIDIA GPU with compute capability 5.0 or higher (Maxwell architecture or newer)
- For sparse matrix support: R package `Matrix`

## Installation

### From Source

```r
# Install dependencies
install.packages("remotes")
install.packages("Matrix") # For sparse matrix support

# Install RidgeRegCuda
remotes::install_github("yourusername/RidgeRegCuda")
```

### Prerequisites

1. Ensure CUDA Toolkit is installed (version 10.0 or higher)
2. Ensure compatible NVIDIA GPU drivers are installed
3. Set the `CUDA_HOME` environment variable to point to your CUDA installation (optional)

## Usage

### Basic Example

```r
library(RidgeRegCuda)

# Check CUDA availability
check_cuda_available()

# List available GPU devices
get_cuda_devices()

# Generate sample data
n_genes <- 5000
n_features <- 100
n_samples <- 20

X <- matrix(rnorm(n_genes * n_features), nrow = n_genes, ncol = n_features)
Y <- matrix(rnorm(n_genes * n_samples), nrow = n_genes, ncol = n_samples)

# Run ridge regression with permutation test (default)
result <- ridge_cuda(X, Y, lambda = 1.0, n_rand = 1000)

# Display summary
summary(result)

# Run ridge regression with t-test
result_ttest <- ridge_cuda(X, Y, lambda = 1.0, n_rand = 0)

# Clean up CUDA resources when done
cleanup_cuda()
```

### Using Sparse Matrices

```r
library(RidgeRegCuda)
library(Matrix)

# Generate sample data
n_genes <- 5000
n_features <- 100
n_samples <- 20

X <- matrix(rnorm(n_genes * n_features), nrow = n_genes, ncol = n_features)
Y <- matrix(rnorm(n_genes * n_samples), nrow = n_genes, ncol = n_samples)

# Create sparse Y matrix (with 95% sparsity)
Y_sparse <- Matrix(Y * (runif(length(Y)) > 0.95), sparse = TRUE)

# Run ridge regression with sparse Y matrix
result_sparse <- ridge_cuda(X, Y_sparse, lambda = 1.0, n_rand = 1000)

summary(result_sparse)
```

## Performance

The CUDA implementation provides significant performance improvements over CPU-based implementations, especially for large datasets and when performing permutation testing with a high number of permutations.

## Troubleshooting

### Common Issues

1. **CUDA not found during installation**: 
   - Ensure CUDA Toolkit is installed
   - Set the `CUDA_HOME` environment variable to your CUDA installation path

2. **Package loads but fails when running functions**:
   - Make sure your NVIDIA drivers are up to date
   - Check GPU compatibility (compute capability 5.0+)
   - Try running with fewer dimensions or permutations

3. **Memory errors**:
   - Reduce the size of input matrices
   - Reduce the number of permutations
   - Use sparse matrices when appropriate

## Citation

If you use this package in your research, please cite:

```
@misc{RidgeRegCuda2025,
  author = {Your Name},
  title = {RidgeRegCuda: CUDA-Accelerated Ridge Regression for R},
  year = {2025},
  url = {https://github.com/yourusername/RidgeRegCuda}
}
```

## License

GPL (>= 2)
