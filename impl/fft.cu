#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "cudautils.cuh"
#include "data.hpp"

__global__ void multiplyComplexKernel(float2 *p1, float2 *p2, float2 *res, int size)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= size)
        return;

    res[tid].x = p1[tid].x * p2[tid].x - p1[tid].y * p2[tid].y;
    res[tid].y = (p1[tid].y) * p2[tid].x + p1[tid].x * (p2[tid].y);
}

__global__ void normalize(float2 *p, int size)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= size)
        return;
    p[tid].x /= size;
}

// Function to perform polynomial multiplication using CUFFT
std::vector<float2> polynomial_multiply(const std::vector<float> &poly1, const std::vector<float> &poly2)
{
    int n = std::max(poly1.size(), poly2.size());
    int res_size = poly1.size() + poly2.size() - 1;
    int padded_size = 1 << (int)ceil(log2(res_size));

    // Allocate device memory for input polynomials and output result
    float2 *d_poly1;
    float2 *d_poly2;
    float2 *d_result;
    cudaCheckErrors(cudaMalloc(&d_poly1, padded_size * sizeof(float2)));
    cudaCheckErrors(cudaMalloc(&d_poly2, padded_size * sizeof(float2)));
    cudaCheckErrors(cudaMalloc(&d_result, padded_size * sizeof(float2)));

    // Copy input polynomials to device memory, padding with zeros
    std::vector<float2> padded_poly1(padded_size, {0.0, 0.0});
    for (int i = 0; i < poly1.size(); i++)
        padded_poly1[i].x = poly1[i];
    std::vector<float2> padded_poly2(padded_size, {0.0, 0.0});
    for (int i = 0; i < poly2.size(); i++)
        padded_poly2[i].x = poly2[i];
    cudaCheckErrors(cudaMemcpy(d_poly1, padded_poly1.data(), padded_size * sizeof(float2), cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(d_poly2, padded_poly2.data(), padded_size * sizeof(float2), cudaMemcpyHostToDevice));

    // Create CUFFT plan for forward and inverse transforms
    cufftHandle plan_forward, plan_inverse;
    cufftPlan1d(&plan_forward, padded_size, CUFFT_C2C, 2);
    cufftPlan1d(&plan_inverse, padded_size, CUFFT_C2C, 2);
    CUDACHKERR();
    // Perform forward FFT on both polynomials
    cufftExecC2C(plan_forward, d_poly1, d_poly1, CUFFT_FORWARD);
    cufftExecC2C(plan_forward, d_poly2, d_poly2, CUFFT_FORWARD);
    CUDACHKERR();

    int TB = 256;
    multiplyComplexKernel<<<std::ceil((float)padded_size / (float)TB), TB>>>(d_poly1, d_poly2, d_result, padded_size);
    CUDACHKERR();
    cufftExecC2C(plan_inverse, d_result, d_result, CUFFT_INVERSE);
    CUDACHKERR();
    normalize<<<std::ceil((float)res_size / (float)TB), TB>>>(d_result, padded_size);

    std::vector<float2> result(res_size);
    cudaCheckErrors(cudaMemcpy(result.data(), d_result, sizeof(float2) * res_size, cudaMemcpyDeviceToHost));

    // Destroy CUFFT plans and free device memory
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    cudaFree(d_poly1);
    cudaFree(d_poly2);
    cudaFree(d_result);

    return result;
}

std::vector<int32_t> string_match_fft(char *text, size_t text_size, char *pattern, size_t pattern_size)
{
    // do all the precomputations

    // convert character arrays into floating point
    std::vector<float> text_coeffs, pattern_coeffs;
    for (int i = 0; i < text_size; i++)
    {
        text_coeffs.push_back((float)text[i]);
    }
    for (int i = 0; i < pattern_size; i++)
    {
        pattern_coeffs.push_back((float)pattern[pattern_size - 1 - i]);
    }
    float pattern_sq = 0.;
    for (auto p : pattern_coeffs)
        pattern_sq += pow(p, 2);
    std::vector<float> prefix_text_sq(text_size + 1, 0.);
    for (int i = 0; i < text_size; i++)
    {
        prefix_text_sq[i + 1] = prefix_text_sq[i] + pow(text_coeffs[i], 2);
    }
    std::vector<float2> pattern_convolution = polynomial_multiply(text_coeffs, pattern_coeffs);

    int start_idx = pattern_size - 1;
    std::vector<int32_t> res;

    for (int i = 0; i < (text_size - pattern_size + 1); i++)
    {
        if (prefix_text_sq[i + pattern_size] - prefix_text_sq[i] + pattern_sq - 2 * pattern_convolution[start_idx + i].x == 0)
            res.push_back(i);
    }

    return res;
}

int main(int argc, char *argv[])
{
    // std::vector<float> poly1 = {1, 2, 3, 4, 5, 6};
    // std::vector<float> poly2 = {4, 5, 6};

    // std::vector<float2> result = polynomial_multiply(poly1, poly2);

    // std::cout << "Result: \n";
    // auto padded_size = 1 << (int)ceil(log2(result.size()));
    // for (int i = 0; i < result.size(); ++i)
    // {
    //     std::cout << result[i].x << "\n";
    // }

    if (argc < 3)
    {
        std::cout << "Please provide path to string column file and pattern. eg: ./brute-force /media/db/comments.txt <like-pattern>";
    }
    std::string txt_file = argv[1];

    gpulike::StringColumn *comments_column = gpulike::read_txt(txt_file);
    char *text = "pending foxes. slyly re";

    size_t text_size = 23;
    // for (int i = 0; i < comments_column->size; i++)
    //     text_size += comments_column->sizes[i];
    char *pattern = "lyl";
    size_t pattern_size = 3;
    auto res = string_match_fft(text, text_size, pattern, pattern_size);
    std::cout << res.size() << " matches found\n";
    return 0;
}

/**
 * p = p1, p2, p3...pm
 * t = t1, t2, ... tn
 *
 *
 */