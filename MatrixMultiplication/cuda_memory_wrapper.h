#pragma once

#include <cuda_runtime.h>

template<typename T>
class CudaMemoryWrapper
{
public:
     CudaMemoryWrapper() : ptr_(nullptr)
     {
     }

     CudaMemoryWrapper(CudaMemoryWrapper<T> const &) = delete;
     CudaMemoryWrapper<T> &operator=(CudaMemoryWrapper<T> const &) = delete;

     T **operator&()
     {
          cudaFree(ptr_);
          return &ptr_;
     }

     T const *get() const
     {
          return ptr_;
     }

     T *get()
     {
          return ptr_;
     }

     ~CudaMemoryWrapper()
     {
          cudaFree(ptr_);
     }

private:
     T *ptr_;
};
