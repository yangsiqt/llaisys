// rms_norm_nvidia.cu — warp-shuffle reduction (faster than shared-memory tree)
#include "rms_norm_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

template <typename T>
__device__ inline float to_f(T v) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(v);
    else if constexpr (std::is_same_v<T, __half>)   return __half2float(v);
    else return (float)v;
}
template <typename T>
__device__ inline T from_f(float v) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) return __float2bfloat16(v);
    else if constexpr (std::is_same_v<T, __half>)   return __float2half(v);
    else return (T)v;
}

template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight,
                                float eps, size_t hidden_size) {
    size_t row = blockIdx.x;
    const T *ri = in + row * hidden_size;
    T *ro = out + row * hidden_size;
    unsigned tid = threadIdx.x;

    float local_sum = 0.f;
    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        float v = to_f(ri[i]);
        local_sum += v * v;
    }

    // Warp-shuffle reduction
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

    __shared__ float warp_sums[32];
    if (tid % 32 == 0) warp_sums[tid / 32] = local_sum;
    __syncthreads();

    if (tid < 32) {
        int nw = (blockDim.x + 31) / 32;
        float val = (tid < (unsigned)nw) ? warp_sums[tid] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (tid == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms_inv = rsqrtf(warp_sums[0] / hidden_size + eps);

    for (size_t i = tid; i < hidden_size; i += blockDim.x)
        ro[i] = from_f<T>(to_f(weight[i]) * to_f(ri[i]) * rms_inv);
}

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              llaisysDataType_t type, size_t batch_size, size_t hidden_size) {
    int t = 256;
    switch (type) {
    case LLAISYS_DTYPE_F32:  rms_norm_kernel<float><<<batch_size,t>>>((float*)out,(const float*)in,(const float*)weight,eps,hidden_size); break;
    case LLAISYS_DTYPE_BF16: rms_norm_kernel<__nv_bfloat16><<<batch_size,t>>>((__nv_bfloat16*)out,(const __nv_bfloat16*)in,(const __nv_bfloat16*)weight,eps,hidden_size); break;
    case LLAISYS_DTYPE_F16:  rms_norm_kernel<__half><<<batch_size,t>>>((__half*)out,(const __half*)in,(const __half*)weight,eps,hidden_size); break;
    default: throw std::runtime_error("Unsupported dtype for CUDA rms_norm");
    }
}
} // namespace llaisys::ops::nvidia
