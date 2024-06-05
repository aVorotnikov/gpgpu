#pragma once

#include <cuda_runtime.h>
#include <string>

void CudaErrorToException(const cudaError_t error, const std::string &file, const std::uint64_t line);

#define CHECK_FAILED(error) (CudaErrorToException(error, __FILE__, __LINE__))
