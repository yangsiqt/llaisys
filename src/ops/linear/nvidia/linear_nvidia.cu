// linear_nvidia.cu — cuBLAS with Tensor Core acceleration + TF32 fast path
#include "linear_nvidia.hpp"
#include "../../../device/nvidia/nvidia_resource.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>

__global__ void add_bias_f32(float *out, const float *bias, size_t n, size_t cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += bias[i % cols];
}
__global__ void add_bias_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *bias, size_t n, size_t cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(out[i]) + __bfloat162float(bias[i % cols]));
}
__global__ void add_bias_f16(__half *out, const __half *bias, size_t n, size_t cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(out[i]) + __half2float(bias[i % cols]));
}

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features) {
    cublasHandle_t handle = llaisys::device::nvidia::getCublasHandle();
    int m = (int)out_features, n = (int)batch_size, k = (int)in_features;
    int thr = 256;
    size_t elems = batch_size * out_features;

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        float alpha = 1.f, beta = 0.f;
        // TF32 Tensor Core: ~3x faster than pure FP32 on Ampere, negligible accuracy loss
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                     weight, CUDA_R_32F, k,
                     in,     CUDA_R_32F, k,
                     &beta,
                     out,    CUDA_R_32F, m,
                     CUBLAS_COMPUTE_32F_FAST_TF32,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (bias)
            add_bias_f32<<<((int)elems + thr - 1) / thr, thr>>>(
                (float *)out, (const float *)bias, elems, out_features);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        float alpha = 1.f, beta = 0.f;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                     weight, CUDA_R_16BF, k,
                     in,     CUDA_R_16BF, k,
                     &beta,
                     out,    CUDA_R_16BF, m,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (bias)
            add_bias_bf16<<<((int)elems + thr - 1) / thr, thr>>>(
                (__nv_bfloat16 *)out, (const __nv_bfloat16 *)bias, elems, out_features);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        float alpha = 1.f, beta = 0.f;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                     weight, CUDA_R_16F, k,
                     in,     CUDA_R_16F, k,
                     &beta,
                     out,    CUDA_R_16F, m,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (bias)
            add_bias_f16<<<((int)elems + thr - 1) / thr, thr>>>(
                (__half *)out, (const __half *)bias, elems, out_features);
        break;
    }
    default: throw std::runtime_error("Unsupported dtype for CUDA linear");
    }
}
} // namespace llaisys::ops::nvidia
