#' Set CUDA Memory Management Options
#'
#' Configures CUDA memory pooling behavior for the RidgeRegCuda package.
#' Memory pooling can improve performance for repeated calculations by
#' reducing memory fragmentation and system allocation overhead.
#'
#' @param enable_pool Logical; whether to enable CUDA memory pooling.
#'                   Default is TRUE.
#' @param allocation_size Initial memory pool size in megabytes (MB).
#'                       Use 0 for CUDA default sizing. Default is 0.
#' @param release_threshold Threshold (in MB) for when memory should be
#'                         released back to the system. Use 0 for CUDA
#'                         default behavior. Default is 0.
#' @return Invisibly returns the previous pool state (TRUE or FALSE).
#' @note This function requires CUDA to be initialized first with
#'       \code{check_cuda_available()}.
#' @examples
#' \dontrun{
#' if (check_cuda_available()$available) {
#'   # Enable memory pooling with default pool size
#'   set_cuda_memory_options(TRUE)
#'   
#'   # Enable with 1GB initial pool and 512MB release threshold
#'   set_cuda_memory_options(TRUE, 1024, 512)
#'   
#'   # Disable memory pooling
#'   set_cuda_memory_options(FALSE)
#' }
#' }
#' @export
#' @useDynLib RidgeRegCuda, .registration = TRUE
set_cuda_memory_options <- function(enable_pool = TRUE, 
                                  allocation_size = 0, 
                                  release_threshold = 0) {
  # Check inputs
  if (!is.logical(enable_pool) || length(enable_pool) != 1) {
    stop("enable_pool must be a single logical value")
  }
  if (!is.numeric(allocation_size) || length(allocation_size) != 1 || allocation_size < 0) {
    stop("allocation_size must be a single non-negative numeric value (in MB)")
  }
  if (!is.numeric(release_threshold) || length(release_threshold) != 1 || release_threshold < 0) {
    stop("release_threshold must be a single non-negative numeric value (in MB)")
  }
  
  # Call C++ function
  tryCatch({
    previous_state <- .Call("ridge_cuda_set_memory_options_r", 
                          as.logical(enable_pool),
                          as.double(allocation_size),
                          as.double(release_threshold),
                          PACKAGE = "RidgeRegCuda")
  }, error = function(e) {
    stop("Failed to set CUDA memory options: ", e$message)
  })
  
  # Print success message
  if (enable_pool) {
    pool_size_str <- if (allocation_size > 0) paste0(allocation_size, " MB") else "default size"
    threshold_str <- if (release_threshold > 0) paste0(release_threshold, " MB") else "default"
    message("CUDA memory pooling enabled with ", pool_size_str, " initial allocation and ", 
            threshold_str, " release threshold.")
  } else {
    message("CUDA memory pooling disabled.")
  }
  
  invisible(previous_state)
}

#' Set CUDA Asynchronous Execution Mode
#'
#' Controls whether CUDA operations execute asynchronously or synchronously.
#' Asynchronous mode can improve performance in some cases by overlapping
#' computation and data transfers, but may make debugging more difficult.
#'
#' @param enable_async Logical; whether to enable asynchronous execution mode.
#'                    Default is TRUE.
#' @return Invisibly returns the previous asynchronous mode state (TRUE or FALSE).
#' @note This function requires CUDA to be initialized first with
#'       \code{check_cuda_available()}.
#' @examples
#' \dontrun{
#' if (check_cuda_available()$available) {
#'   # Enable asynchronous execution
#'   set_cuda_async_mode(TRUE)
#'   
#'   # Disable asynchronous execution (synchronous mode)
#'   set_cuda_async_mode(FALSE)
#' }
#' }
#' @export
#' @useDynLib RidgeRegCuda, .registration = TRUE
set_cuda_async_mode <- function(enable_async = TRUE) {
  # Check input
  if (!is.logical(enable_async) || length(enable_async) != 1) {
    stop("enable_async must be a single logical value")
  }
  
  # Call C++ function
  tryCatch({
    previous_mode <- .Call("ridge_cuda_set_async_mode_r", 
                          as.logical(enable_async),
                          PACKAGE = "RidgeRegCuda")
  }, error = function(e) {
    stop("Failed to set CUDA asynchronous mode: ", e$message)
  })
  
  # Print success message
  if (enable_async) {
    message("CUDA asynchronous execution mode enabled.")
  } else {
    message("CUDA asynchronous execution mode disabled (synchronous mode).")
  }
  
  invisible(previous_mode)
}