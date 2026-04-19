#' Get Available CUDA Devices
#'
#' Returns information about available CUDA devices detected on the system.
#'
#' @param max_devices Deprecated argument, no longer used. The function now
#'   queries all available devices.
#' @return A data frame with columns:
#'   \describe{
#'     \item{device_id}{Integer ID of the device (0-based).}
#'     \item{name}{Character string with the device name.}
#'     \item{memory_mb}{Numeric value of total global memory in Megabytes (MB).}
#'     \item{memory_gb}{Numeric value of total global memory in Gigabytes (GB).}
#'   }
#'   Returns an empty data frame if no devices are found or if an error occurs
#'   during query.
#' @examples
#' \dontrun{
#' devices <- get_cuda_devices()
#' print(devices)
#' if (nrow(devices) > 0) {
#'   # Select the device with the most memory
#'   best_device_id <- devices$device_id[which.max(devices$memory_mb)]
#'   print(paste("Using device:", best_device_id))
#' }
#' }
#' @export
#' @useDynLib RidgeCuda, .registration = TRUE
get_cuda_devices <- function(max_devices = NULL) {
   # Issue deprecation warning if max_devices is used
   if (!is.null(max_devices)) {
      warning("'max_devices' argument is deprecated in get_cuda_devices() and is ignored.")
   }

  # Call the C++ interface function registered as "get_cuda_devices_r"
  devices <- .Call("get_cuda_devices_r", as.integer(1), PACKAGE = "RidgeCuda") # Pass dummy integer

  # Post-process the result
  if (is.data.frame(devices) && nrow(devices) > 0 && "memory_mb" %in% names(devices)) {
      devices$memory_gb <- round(devices$memory_mb / 1024, 2)
  } else if (is.data.frame(devices) && nrow(devices) == 0) {
      # Ensure empty df has the GB column too
      devices$memory_gb <- numeric(0)
  } else {
      warning("Could not retrieve valid device information. Returning empty data frame.")
      # Construct a correctly structured empty data frame
      devices <- data.frame(device_id = integer(0),
                            name = character(0),
                            memory_mb = numeric(0),
                            memory_gb = numeric(0),
                            stringsAsFactors = FALSE)
  }

  return(devices)
}
