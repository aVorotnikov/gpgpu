#include "matrix.cuh"
#include "matrix_utils.cuh"
#include "cuda_error_wrapper.h"

#include <iostream>
#include <chrono>

int main()
try
{
    static constexpr const int size = 1024;

    std::cout << "Method,Time,Correctness" << std::endl;

    Matrix a = GetRandomMatrix(size, size);
    Matrix b = GetRandomMatrix(size, size);

    auto start = std::chrono::steady_clock::now();
    Matrix resCpu = a.Mul(b, Matrix::MultiplicationMode::CPU);
    auto end = std::chrono::steady_clock::now();
    auto dif = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "CPU," << dif << "ms,1" << std::endl;

    start = std::chrono::steady_clock::now();
    Matrix resSimple = a.Mul(b, Matrix::MultiplicationMode::SIMPLE);
    end = std::chrono::steady_clock::now();
    dif = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Simple," << dif << "ms," << CheckEquality(resCpu, resSimple, size, size) << std::endl;

    start = std::chrono::steady_clock::now();
    Matrix resShared = a.Mul(b, Matrix::MultiplicationMode::SHARED);
    end = std::chrono::steady_clock::now();
    dif = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Shared," << dif << "ms," << CheckEquality(resCpu, resShared, size, size) << std::endl;

    start = std::chrono::steady_clock::now();
    Matrix resIntrinsics = a.Mul(b, Matrix::MultiplicationMode::INTRINSICS);
    end = std::chrono::steady_clock::now();
    dif = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Intrinsics," << dif << "ms," << CheckEquality(resCpu, resIntrinsics, size, size) << std::endl;

    CHECK_FAILED(cudaDeviceReset());

    return EXIT_SUCCESS;
}
catch (std::exception &e)
{
     std::cout << e.what();
     return EXIT_FAILURE;
}
