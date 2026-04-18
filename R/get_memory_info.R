#' Get Current CUDA Device Memory Info
#'
#' Retrieves free/total memory for the currently active CUDA device. Requires
#' that CUDA has been successfully initialized (e.g., via `check_cuda_available`).
#'
#' @return A list containing `free_memory` and `total_memory` (bytes).
#' @examples
#' \dontrun{
#' if(check_cuda_available()$available) {
#'   mem_info <- get_cuda_memory_info()
#'   print(paste("Free:", round(mem_info$free_memory / 1024^3, 2), "GB"))
#'   print(paste("Total:", round(mem_info$total_memory / 1024^3, 2), "GB"))
#' }
#' }
#' @export
#' @useDynLib RidgeRegCuda, .registration = TRUE
get_cuda_memory_info <- function() {
   # Ensure CUDA is initialized before querying. Check default device 0.
   # User should ideally call check_cuda_available first if using non-default device.
   # Use tryCatch to handle potential errors during the check itself
   init_stat <- tryCatch(
       .Call("check_cuda_available_r", as.integer(0), PACKAGE = "RidgeRegCuda"),
       error = function(e) list(status = -99, message = paste("Error during CUDA check:", e$message))
   )

   if (!is.list(init_stat) || is.null(init_stat$status) || init_stat$status != 0) {
       msg <- if (is.list(init_stat) && !is.null(init_stat$message)) init_stat$message else "Unknown CUDA check/initialization error"
       stop("CUDA check/initialization failed for device 0, cannot get memory info. Message: ", msg)
   }

   # Call C++ function registered as "ridge_cuda_get_memory_info_r"
   result <- .Call("ridge_cuda_get_memory_info_r", as.integer(0), PACKAGE = "RidgeRegCuda") # Pass dummy device_id

   # Convert output memory values to numeric
   result$free_memory <- as.numeric(result$free_memory)
   result$total_memory <- as.numeric(result$total_memory)
   return(result)
}
