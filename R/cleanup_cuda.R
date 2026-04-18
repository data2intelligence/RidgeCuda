#' Clean Up CUDA Resources
#'
#' Releases CUDA resources associated with the RidgeRegCuda package. It's good
#' practice to call this when you are finished with GPU computations, although
#' resources are typically released when the R session ends. Also called automatically
#' by `.onUnload`.
#'
#' @return Invisibly returns `TRUE`.
#' @examples
#' \dontrun{
#' # ... perform ridge_cuda computations ...
#' cleanup_cuda()
#' }
#' @export
#' @useDynLib RidgeRegCuda, .registration = TRUE
cleanup_cuda <- function() {
  # Call C++ function registered as "cleanup_cuda_r"
  invisible(.Call("cleanup_cuda_r", PACKAGE = "RidgeRegCuda"))
}
