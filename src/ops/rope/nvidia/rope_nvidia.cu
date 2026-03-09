#include "rope_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

// Each thread handles one (s, h, j) pair where j < head_dim/2
template<typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                            float theta, size_t seq_len, size_t n_heads, size_t head_dim) {
    size_t half_dim = head_dim / 2;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * n_heads * half_dim) return;

    size_t j = idx % half_dim;
    size_t h = (idx / half_dim) % n_heads;
    size_t s = idx / (half_dim * n_heads);

    float pos = (float)pos_ids[s];
    float freq = pos / powf(theta, (2.f * j) / head_dim);
    float cos_f = cosf(freq), sin_f = sinf(freq);

    size_t base = s * n_heads * head_dim + h * head_dim;
    float a, b;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        a = __bfloat162float(in[base + j]); b = __bfloat162float(in[base + j + half_dim]);
        out[base + j]           = __float2bfloat16(a*cos_f - b*sin_f);
        out[base + j + half_dim]= __float2bfloat16(b*cos_f + a*sin_f);
    } else if constexpr (std::is_same_v<T, __half>) {
        a = __half2float(in[base + j]); b = __half2float(in[base + j + half_dim]);
        out[base + j]           = __float2half(a*cos_f - b*sin_f);
        out[base + j + half_dim]= __float2half(b*cos_f + a*sin_f);
    } else {
        a = in[base + j]; b = in[base + j + half_dim];
        out[base + j]           = (T)(a*cos_f - b*sin_f);
        out[base + j + half_dim]= (T)(b*cos_f + a*sin_f);
    }
}

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
          llaisysDataType_t type, size_t seq_len, size_t n_heads, size_t head_dim) {
    size_t total = seq_len * n_heads * (head_dim / 2);
    int t = 256, b = (total + t - 1) / t;
    switch (type) {
    case LLAISYS_DTYPE_F32:  rope_kernel<float><<<b,t>>>((float*)out,(const float*)in,(const int64_t*)pos_ids,theta,seq_len,n_heads,head_dim); break;
    case LLAISYS_DTYPE_BF16: rope_kernel<__nv_bfloat16><<<b,t>>>((__nv_bfloat16*)out,(const __nv_bfloat16*)in,(const int64_t*)pos_ids,theta,seq_len,n_heads,head_dim); break;
    case LLAISYS_DTYPE_F16:  rope_kernel<__half><<<b,t>>>((__half*)out,(const __half*)in,(const int64_t*)pos_ids,theta,seq_len,n_heads,head_dim); break;
    default: throw std::runtime_error("Unsupported dtype for CUDA rope");
    }
}
} // namespace llaisys::ops::nvidia
