#include "matrix.cuh"
#include "cuda_error_wrapper.h"
#include "cuda_memory_wrapper.h"

#include <device_launch_parameters.h>

#include <stdexcept>

namespace
{
     constexpr const std::size_t warpSize = 32;

     void MatrixMultiplicationCpu(
          float const *a, float const *b, float *c,
          const std::size_t l, const std::size_t m, const std::size_t n)
     {
          for (std::size_t i = 0; i < l; ++i)
               for (std::size_t j = 0; j < n; ++j)
               {
                    c[i * n + j] = 0;
                    for (std::size_t k = 0; k < m; ++k)
                         c[i * n + j] += a[i * m + k] * b[k * n + j];
               }
     }

     __global__ void MatrixMultiplicationSimple(
          float const *a, float const *b, float *c,
          const std::size_t l, const std::size_t m, const std::size_t n)
     {
          const std::size_t cCol = blockIdx.x * blockDim.x + threadIdx.x;
          const std::size_t cRow = blockIdx.y * blockDim.y + threadIdx.y;

          if (cRow >= l || cCol >= n)
               return;

          float sum = 0;
          for (std::size_t i = 0; i < m; ++i)
               sum += a[cRow * m + i] * b[i * n + cCol];
          c[cRow * n + cCol] = sum;
     }

     __global__ void MatrixMultiplicationShared(
          float const *a, float const *b, float *c,
          const std::size_t l, const std::size_t m, const std::size_t n)
     {
          const std::size_t cCol = blockIdx.x * blockDim.x + threadIdx.x;
          const std::size_t cRow = blockIdx.y * blockDim.y + threadIdx.y;
          const std::size_t tileCol = threadIdx.x;
          const std::size_t tileRow = threadIdx.y;

          __shared__ float aTile[warpSize][warpSize];
          __shared__ float bTile[warpSize][warpSize + 1];

          float cVal = 0.f;
          const bool isOutOfC = cRow >= l || cCol >= n;

          for (std::size_t tileId = 0; tileId < (m - 1) / warpSize + 1; ++tileId)
          {
               aTile[tileRow][tileCol] = !isOutOfC ? a[cRow * m + (tileId * warpSize + tileCol)] : 0.f;
               bTile[tileRow][tileCol] = !isOutOfC ? b[(tileId * warpSize + tileRow) * n + cCol] : 0.f;
               __syncthreads();

               for (std::size_t i = 0; i < warpSize; ++i)
                    cVal += aTile[tileRow][i] * bTile[i][tileCol];
               __syncthreads();
          }
          if (!isOutOfC)
               c[cRow * n + cCol] = cVal;
     }

     __global__ void MatrixMultiplicationIntrinsics(
          float const *a, float const *b, float *c,
          const std::size_t l, const std::size_t m, const std::size_t n)
     {
          const std::size_t cCol = blockIdx.x * blockDim.x + threadIdx.x;
          const std::size_t cRow = blockIdx.y * blockDim.y + threadIdx.y;
          const std::size_t tileCol = threadIdx.x;
          const std::size_t tileRow = threadIdx.y;

          __shared__ float aTile[warpSize][warpSize];
          __shared__ float bTile[warpSize][warpSize + 1];

          float cVal = 0.f;
          const bool isOutOfC = cRow >= l || cCol >= n;

          for (std::size_t tileId = 0; tileId < (m - 1) / warpSize + 1; ++tileId)
          {
               aTile[tileRow][tileCol] = !isOutOfC ? a[cRow * m + (tileId * warpSize + tileCol)] : 0.f;
               bTile[tileRow][tileCol] = !isOutOfC ? b[(tileId * warpSize + tileRow) * n + cCol] : 0.f;
               __syncthreads();

               float aTileLocal = aTile[tileRow][tileCol];
               __syncwarp();
               for (std::size_t i = 0; i < warpSize; ++i)
                    cVal += __shfl_sync(0xffffffff, aTileLocal, i) * bTile[i][tileCol];
               __syncthreads();
          }
          if (!isOutOfC)
               c[cRow * n + cCol] = cVal;
     }

     void MatrixMultiplication(
          float const *a, float const *b, float *c,
          const std::size_t l, const std::size_t m, const std::size_t n,
          const Matrix::MultiplicationMode mode)
     {
          if (Matrix::MultiplicationMode::CPU == mode)
          {
               MatrixMultiplicationCpu(a, b, c, l, m, n);
               return;
          }

          CHECK_FAILED(cudaSetDevice(0));

          CudaMemoryWrapper<float> aDevice, bDevice, cDevice;
          CHECK_FAILED(cudaMalloc(&aDevice, l * m * sizeof(float)));
          CHECK_FAILED(cudaMalloc(&bDevice, m * n * sizeof(float)));
          CHECK_FAILED(cudaMalloc(&cDevice, l * n * sizeof(float)));
          CHECK_FAILED(
               cudaMemcpy(aDevice.get(), a, l * m * sizeof(float), cudaMemcpyHostToDevice));
          CHECK_FAILED(
               cudaMemcpy(bDevice.get(), b, m * n * sizeof(float), cudaMemcpyHostToDevice));

          dim3 blockInGrid(
               (n - 1ULL) / warpSize + 1ULL,
               (l - 1ULL) / warpSize + 1ULL);
          dim3 threadInBlock(warpSize, warpSize);

          switch (mode)
          {
               case Matrix::MultiplicationMode::SIMPLE:
                    MatrixMultiplicationSimple<<<blockInGrid, threadInBlock>>>(
                         aDevice.get(), bDevice.get(), cDevice.get(), l, m, n);
                    break;
               case Matrix::MultiplicationMode::SHARED:
                    MatrixMultiplicationShared<<<blockInGrid, threadInBlock>>>(
                         aDevice.get(), bDevice.get(), cDevice.get(), l, m, n);
                    break;
               case Matrix::MultiplicationMode::INTRINSICS:
                    MatrixMultiplicationIntrinsics<<<blockInGrid, threadInBlock>>>(
                         aDevice.get(), bDevice.get(), cDevice.get(), l, m, n);
                    break;
          }

          CHECK_FAILED(cudaDeviceSynchronize());
          CHECK_FAILED(
               cudaMemcpy(c, cDevice.get(), l * n * sizeof(float), cudaMemcpyDeviceToHost));
          CHECK_FAILED(cudaGetLastError());
     }
}

Matrix::Matrix(const std::size_t w, const std::size_t h) : w_(w), h_(h), data_(w * h)
{
}

float Matrix::At(const std::size_t i, const std::size_t j) const
{
     return data_[i * w_ + j];
}

float &Matrix::At(const std::size_t i, const std::size_t j)
{
     return data_[i * w_ + j];
}

Matrix Matrix::Mul(Matrix const &other, const Matrix::MultiplicationMode mode) const
{
     if (w_ != other.h_)
          throw std::invalid_argument("Matrix dimensions mismatch");

     Matrix res(h_, other.w_);
     MatrixMultiplication(
          data_.data(), other.data_.data(), res.data_.data(),
          h_, w_, other.w_, mode);

     return res;
}
