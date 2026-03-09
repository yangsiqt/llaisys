#include "swiglu_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

template<typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g, u;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) { g=__bfloat162float(gate[idx]); u=__bfloat162float(up[idx]); out[idx]=__float2bfloat16(u*g/(1.f+expf(-g))); }
    else if constexpr (std::is_same_v<T, __half>)   { g=__half2float(gate[idx]);     u=__half2float(up[idx]);     out[idx]=__float2half(u*g/(1.f+expf(-g))); }
    else { g=(float)gate[idx]; u=(float)up[idx]; out[idx]=(T)(u*g/(1.f+expf(-g))); }
}

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel) {
    int t=256, b=(numel+t-1)/t;
    switch (type) {
    case LLAISYS_DTYPE_F32:  swiglu_kernel<float><<<b,t>>>((float*)out,(const float*)gate,(const float*)up,numel); break;
    case LLAISYS_DTYPE_BF16: swiglu_kernel<__nv_bfloat16><<<b,t>>>((__nv_bfloat16*)out,(const __nv_bfloat16*)gate,(const __nv_bfloat16*)up,numel); break;
    case LLAISYS_DTYPE_F16:  swiglu_kernel<__half><<<b,t>>>((__half*)out,(const __half*)gate,(const __half*)up,numel); break;
    default: throw std::runtime_error("Unsupported dtype for CUDA swiglu");
    }
}
} // namespace llaisys::ops::nvidia
