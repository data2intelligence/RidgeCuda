#' On Package Unload Hook
#'
#' Cleans up CUDA resources when the package is unloaded.
#'
#' @param libpath Library path (unused).
#' @keywords internal
#' @importFrom utils packageVersion
#' @importFrom methods as 
.onUnload <- function(libpath) {
  # Try to cleanup, but don't error if it fails (e.g., if context was already lost)
  try(cleanup_cuda(), silent = TRUE)
  # Standard unload procedure
  library.dynam.unload("RidgeRegCuda", libpath)
}
