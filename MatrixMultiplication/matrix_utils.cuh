#pragma once

#include "matrix.cuh"

Matrix GetRandomMatrix(
     const std::size_t w, const std::size_t h,
     const float down = -10.0f, const float up = -10.0f);

bool CheckEquality(
     const Matrix a, const Matrix b,
     const std::size_t w, const std::size_t h,
     const float treshold = 1e-5);
