#include "cuda_error_wrapper.h"

#include <stdexcept>

void CudaErrorToException(const cudaError_t error, const std::string &file, const std::uint64_t line)
{
     if (cudaSuccess != error)
     {
          std::string what =
               std::string("CUDA error: ") + cudaGetErrorString(error) +
               " (" + file + ":" + std::to_string(line) + ")";
          throw std::runtime_error(what);
     }
}
