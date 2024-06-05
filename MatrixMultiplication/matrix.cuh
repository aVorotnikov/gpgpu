#pragma once

#include <vector>

class Matrix
{
public:
     enum class MultiplicationMode
     {
          CPU,
          SIMPLE,
          SHARED,
          INTRINSICS
     };

     Matrix(const std::size_t w, const std::size_t h);

     size_t Height() const
     {
          return h_;
     }
     size_t Width() const
     {
          return w_;
     }

     float At(const std::size_t i, const std::size_t j) const;
     float &At(const std::size_t i, const std::size_t j);

     Matrix Mul(Matrix const &other, const MultiplicationMode mode) const;

private:
     std::size_t w_, h_;
     std::vector<float> data_;
};
