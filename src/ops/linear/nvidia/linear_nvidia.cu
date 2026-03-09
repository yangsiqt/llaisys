#include "linear_nvidia.hpp"
#include "../../../device/nvidia/nvidia_resource.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>

// Add bias kernels
__global__ void add_bias_f32(float *out, const float *bias, size_t batch, size_t out_f) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * out_f) out[idx] += bias[idx % out_f];
}
__global__ void add_bias_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *bias, size_t batch, size_t out_f) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * out_f) out[idx] = __float2bfloat16(__bfloat162float(out[idx]) + __bfloat162float(bias[idx % out_f]));
}
__global__ void add_bias_f16(__half *out, const __half *bias, size_t batch, size_t out_f) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * out_f) out[idx] = __float2half(__half2float(out[idx]) + __half2float(bias[idx % out_f]));
}

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features) {
    cublasHandle_t handle = llaisys::device::nvidia::getCublasHandle();
    int m = (int)out_features, n = (int)batch_size, k = (int)in_features;

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        float alpha = 1.f, beta = 0.f;
        if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                        (const float*)weight, k, (const float*)in, k, &beta, (float*)out, m)
            != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSgemm failed");
        if (bias) { int t=256; add_bias_f32<<<((int)(batch_size*out_features)+t-1)/t,t>>>((float*)out,(const float*)bias,batch_size,out_features); }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        float alpha = 1.f, beta = 0.f;
        if (cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                         weight, CUDA_R_16BF, k, in, CUDA_R_16BF, k, &beta,
                         out, CUDA_R_16BF, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
            != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGemmEx BF16 failed");
        if (bias) { int t=256; add_bias_bf16<<<((int)(batch_size*out_features)+t-1)/t,t>>>((__nv_bfloat16*)out,(const __nv_bfloat16*)bias,batch_size,out_features); }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        float alpha = 1.f, beta = 0.f;
        if (cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                         weight, CUDA_R_16F, k, in, CUDA_R_16F, k, &beta,
                         out, CUDA_R_16F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
            != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGemmEx F16 failed");
        if (bias) { int t=256; add_bias_f16<<<((int)(batch_size*out_features)+t-1)/t,t>>>((__half*)out,(const __half*)bias,batch_size,out_features); }
        break;
    }
    default: throw std::runtime_error("Unsupported dtype for CUDA linear");
    }
}
} // namespace llaisys::ops::nvidia
