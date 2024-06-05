#include "matrix_utils.cuh"

Matrix GetRandomMatrix(
     const std::size_t w, const std::size_t h,
     const float down, const float up)
{
     auto res = Matrix(w, h);
     const auto dif = up - down;
     for (std::size_t i = 0; i < h; ++i)
          for (std::size_t j = 0; j < w; ++j)
               res.At(i, j) = (1.0f * rand()) / RAND_MAX * dif + down;
     return res;
}

bool CheckEquality(
     const Matrix a, const Matrix b,
     const std::size_t w, const std::size_t h,
     const float treshold)
{
     for (std::size_t i = 0; i < h; ++i)
          for (std::size_t j = 0; j < w; ++j)
               if (std::abs(a.At(i, j) - b.At(i, j)) > treshold)
                    return false;
     return true;
}
