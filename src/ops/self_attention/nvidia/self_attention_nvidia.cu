// self_attention_nvidia.cu — fused kernel, optimal for autoregressive decode
#include "self_attention_nvidia.hpp"
#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename T>
__device__ inline float load_f(const T *p, size_t i) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(p[i]);
    else if constexpr (std::is_same_v<T, __half>)   return __half2float(p[i]);
    else                                              return (float)p[i];
}
template <typename T>
__device__ inline void store_f(T *p, size_t i, float v) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) p[i] = __float2bfloat16(v);
    else if constexpr (std::is_same_v<T, __half>)   p[i] = __float2half(v);
    else                                              p[i] = (T)v;
}

template <typename T>
__global__ void self_attn_kernel(T *attn_val, const T *q, const T *k, const T *v,
                                 float scale, size_t qlen, size_t kvlen,
                                 size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    size_t qi = blockIdx.x / n_heads;
    size_t h  = blockIdx.x % n_heads;
    size_t kv_h = h / (n_heads / n_kv_heads);
    size_t tid = threadIdx.x;
    size_t abs_qi = kvlen - qlen + qi;

    extern __shared__ float scores[];

    for (size_t ki = tid; ki < kvlen; ki += blockDim.x) {
        float sc = 0.f;
        for (size_t d = 0; d < head_dim; d++)
            sc += load_f(q, qi * n_heads * head_dim + h * head_dim + d)
                * load_f(k, ki * n_kv_heads * head_dim + kv_h * head_dim + d);
        scores[ki] = (ki <= abs_qi) ? sc * scale : -FLT_MAX;
    }
    __syncthreads();

    __shared__ float s_max[32], s_sum[32];
    float lmax = -FLT_MAX;
    for (size_t ki = tid; ki < kvlen; ki += blockDim.x) lmax = fmaxf(lmax, scores[ki]);
    for (int off = 16; off > 0; off >>= 1) lmax = fmaxf(lmax, __shfl_down_sync(0xffffffff, lmax, off));
    if (tid % 32 == 0) s_max[tid / 32] = lmax;
    __syncthreads();
    if (tid < 32) {
        int nw = (blockDim.x + 31) / 32;
        lmax = (tid < (size_t)nw) ? s_max[tid] : -FLT_MAX;
        for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_down_sync(0xffffffff, lmax, o));
        if (tid == 0) s_max[0] = lmax;
    }
    __syncthreads();
    float gmax = s_max[0];

    float lsum = 0.f;
    for (size_t ki = tid; ki < kvlen; ki += blockDim.x) {
        scores[ki] = expf(scores[ki] - gmax);
        lsum += scores[ki];
    }
    for (int off = 16; off > 0; off >>= 1) lsum += __shfl_down_sync(0xffffffff, lsum, off);
    if (tid % 32 == 0) s_sum[tid / 32] = lsum;
    __syncthreads();
    if (tid < 32) {
        int nw = (blockDim.x + 31) / 32;
        lsum = (tid < (size_t)nw) ? s_sum[tid] : 0.f;
        for (int o = 16; o > 0; o >>= 1) lsum += __shfl_down_sync(0xffffffff, lsum, o);
        if (tid == 0) s_sum[0] = lsum;
    }
    __syncthreads();
    float gsum = s_sum[0];

    for (size_t ki = tid; ki < kvlen; ki += blockDim.x) scores[ki] /= gsum;
    __syncthreads();

    for (size_t d = tid; d < head_dim; d += blockDim.x) {
        float val = 0.f;
        for (size_t ki = 0; ki < kvlen; ki++)
            val += scores[ki] * load_f(v, ki * n_kv_heads * head_dim + kv_h * head_dim + d);
        store_f(attn_val, qi * n_heads * head_dim + h * head_dim + d, val);
    }
}

namespace llaisys::ops::nvidia {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t type,
                    size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    int blocks = qlen * n_heads, threads = 256;
    size_t smem = kvlen * sizeof(float);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attn_kernel<float><<<blocks, threads, smem>>>((float *)attn_val, (const float *)q, (const float *)k, (const float *)v, scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attn_kernel<__nv_bfloat16><<<blocks, threads, smem>>>((__nv_bfloat16 *)attn_val, (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v, scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        self_attn_kernel<__half><<<blocks, threads, smem>>>((__half *)attn_val, (const __half *)q, (const __half *)k, (const __half *)v, scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA self_attention");
    }
}
} // namespace llaisys::ops::nvidia
