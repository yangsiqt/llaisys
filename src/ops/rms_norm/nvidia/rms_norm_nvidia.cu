#include "rms_norm_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

// One block per row, shared memory reduction
template<typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight,
                                float eps, size_t hidden_size) {
    extern __shared__ float smem[];
    size_t row = blockIdx.x;
    const T *row_in = in + row * hidden_size;
    T *row_out = out + row * hidden_size;

    float local_sum = 0.f;
    for (size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v;
        if constexpr (sizeof(T) == 2) {
            if constexpr (std::is_same_v<T, __nv_bfloat16>) v = __bfloat162float(row_in[i]);
            else v = __half2float(row_in[i]);
        } else { v = row_in[i]; }
        local_sum += v * v;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float rms_inv = rsqrtf(smem[0] / hidden_size + eps);

    for (size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v, w;
        if constexpr (sizeof(T) == 2) {
            if constexpr (std::is_same_v<T, __nv_bfloat16>) { v = __bfloat162float(row_in[i]); w = __bfloat162float(weight[i]); row_out[i] = __float2bfloat16(w * v * rms_inv); }
            else { v = __half2float(row_in[i]); w = __half2float(weight[i]); row_out[i] = __float2half(w * v * rms_inv); }
        } else { row_out[i] = (T)(weight[i] * row_in[i] * rms_inv); }
    }
}

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              llaisysDataType_t type, size_t batch_size, size_t hidden_size) {
    int t = 256; // power-of-2, fits hidden_size up to 256*N
    while (t > 1 && (size_t)t > hidden_size) t >>= 1;
    if ((size_t)t < hidden_size) t = 256; // fallback: use 256 with loop
    size_t smem = t * sizeof(float);
    switch (type) {
    case LLAISYS_DTYPE_F32:  rms_norm_kernel<float><<<batch_size,t,smem>>>((float*)out,(const float*)in,(const float*)weight,eps,hidden_size); break;
    case LLAISYS_DTYPE_BF16: rms_norm_kernel<__nv_bfloat16><<<batch_size,t,smem>>>((__nv_bfloat16*)out,(const __nv_bfloat16*)in,(const __nv_bfloat16*)weight,eps,hidden_size); break;
    case LLAISYS_DTYPE_F16:  rms_norm_kernel<__half><<<batch_size,t,smem>>>((__half*)out,(const __half*)in,(const __half*)weight,eps,hidden_size); break;
    default: throw std::runtime_error("Unsupported dtype for CUDA rms_norm");
    }
}
} // namespace llaisys::ops::nvidia
