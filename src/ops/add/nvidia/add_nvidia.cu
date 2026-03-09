#include "add_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

__global__ void add_f32_kernel(float *out, const float *a, const float *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}
__global__ void add_bf16_kernel(__nv_bfloat16 *out, const __nv_bfloat16 *a, const __nv_bfloat16 *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
}
__global__ void add_f16_kernel(__half *out, const __half *a, const __half *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
}

namespace llaisys::ops::nvidia {
void add(std::byte *out, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel) {
    int threads = 256, blocks = (numel + threads - 1) / threads;
    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_f32_kernel<<<blocks,threads>>>(reinterpret_cast<float*>(out),reinterpret_cast<const float*>(a),reinterpret_cast<const float*>(b),numel); break;
    case LLAISYS_DTYPE_BF16:
        add_bf16_kernel<<<blocks,threads>>>(reinterpret_cast<__nv_bfloat16*>(out),reinterpret_cast<const __nv_bfloat16*>(a),reinterpret_cast<const __nv_bfloat16*>(b),numel); break;
    case LLAISYS_DTYPE_F16:
        add_f16_kernel<<<blocks,threads>>>(reinterpret_cast<__half*>(out),reinterpret_cast<const __half*>(a),reinterpret_cast<const __half*>(b),numel); break;
    default: throw std::runtime_error("Unsupported dtype for CUDA add");
    }
}
} // namespace llaisys::ops::nvidia
