// self_attention_nvidia.cu — FlashAttention-style tiled attention (online softmax over KV blocks).
// One block per (query position, head). KV sequence is processed in tiles to bound shared memory
// and improve HBM reuse vs materializing full [kvlen] score rows.
#include "self_attention_nvidia.hpp"
#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename T>
__device__ inline float load_f(const T *p, size_t i) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(p[i]);
    else if constexpr (std::is_same_v<T, __half>) return __half2float(p[i]);
    else return (float)p[i];
}
template <typename T>
__device__ inline void store_f(T *p, size_t i, float v) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) p[i] = __float2bfloat16(v);
    else if constexpr (std::is_same_v<T, __half>) p[i] = __float2half(v);
    else p[i] = (T)v;
}

__device__ inline float warp_reduce_max(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}

__device__ inline float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ float block_reduce_max(float val) {
    const int wid = threadIdx.x / 32;
    const int nw = (blockDim.x + 31) / 32;
    __shared__ float warp_max[32];
    val = warp_reduce_max(val);
    if ((threadIdx.x & 31) == 0) warp_max[wid] = val;
    __syncthreads();
    val = (threadIdx.x < nw) ? warp_max[threadIdx.x] : -FLT_MAX;
    if (wid == 0) val = warp_reduce_max(val);
    if (threadIdx.x == 0) warp_max[0] = val;
    __syncthreads();
    return warp_max[0];
}

__device__ float block_reduce_sum(float val) {
    const int wid = threadIdx.x / 32;
    const int nw = (blockDim.x + 31) / 32;
    __shared__ float warp_sum[32];
    val = warp_reduce_sum(val);
    if ((threadIdx.x & 31) == 0) warp_sum[wid] = val;
    __syncthreads();
    val = (threadIdx.x < nw) ? warp_sum[threadIdx.x] : 0.f;
    if (wid == 0) val = warp_reduce_sum(val);
    if (threadIdx.x == 0) warp_sum[0] = val;
    __syncthreads();
    return warp_sum[0];
}

// BR = max keys per tile (static upper bound; actual tile may be shorter).
template <typename T, int BR>
__global__ void self_attn_fa2_kernel(T *attn_val, const T *q, const T *k, const T *v, float scale, size_t qlen,
                                     size_t kvlen, size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    const size_t qi = blockIdx.x / n_heads;
    const size_t h = blockIdx.x % n_heads;
    const size_t kv_h = h / (n_heads / n_kv_heads);
    const unsigned tid = threadIdx.x;
    const size_t abs_qi = kvlen - qlen + qi;

    extern __shared__ unsigned char smem_u[];
    float *const K_tile = reinterpret_cast<float *>(smem_u);
    float *const V_tile = K_tile + BR * head_dim;
    float *const S_tile = V_tile + BR * head_dim;
    float *const Q_tile = S_tile + BR;
    float *const O_acc = Q_tile + head_dim;

    for (size_t d = tid; d < head_dim; d += blockDim.x)
        Q_tile[d] = load_f(q, qi * n_heads * head_dim + h * head_dim + d);
    __syncthreads();

    float m = -FLT_MAX;
    float l = 0.f;
    for (size_t d = tid; d < head_dim; d += blockDim.x) O_acc[d] = 0.f;
    __syncthreads();

    for (int k0 = 0; k0 < (int)kvlen; k0 += BR) {
        const int tile_len = min(BR, (int)kvlen - k0);

        for (int idx = tid; idx < tile_len * (int)head_dim; idx += blockDim.x) {
            const int b = idx / (int)head_dim;
            const int d = idx % (int)head_dim;
            const size_t ki = (size_t)k0 + (size_t)b;
            const size_t off = ki * n_kv_heads * head_dim + kv_h * head_dim + (size_t)d;
            K_tile[b * (int)head_dim + d] = load_f(k, off);
            V_tile[b * (int)head_dim + d] = load_f(v, off);
        }
        __syncthreads();

        for (int b = tid; b < tile_len; b += blockDim.x) {
            float sc = 0.f;
            for (size_t d = 0; d < head_dim; d++)
                sc += Q_tile[d] * K_tile[b * (int)head_dim + (int)d];
            sc *= scale;
            const size_t ki = (size_t)k0 + (size_t)b;
            if (ki > abs_qi) sc = -FLT_MAX;
            S_tile[b] = sc;
        }
        __syncthreads();

        float local_max = -FLT_MAX;
        for (int b = tid; b < tile_len; b += blockDim.x) local_max = fmaxf(local_max, S_tile[b]);
        const float m_tile = block_reduce_max(local_max);
        const float m_new = fmaxf(m, m_tile);
        __syncthreads();

        for (int b = tid; b < tile_len; b += blockDim.x) {
            const float s_raw = S_tile[b];
            const float pb = expf(s_raw - m_new);
            S_tile[b] = (s_raw > (-FLT_MAX * 0.5f)) ? pb : 0.f;
        }
        __syncthreads();

        float local_sum = 0.f;
        for (int b = tid; b < tile_len; b += blockDim.x) local_sum += S_tile[b];
        const float sum_p = block_reduce_sum(local_sum);

        const float alpha = expf(m - m_new);
        const float l_new = alpha * l + sum_p;

        for (size_t d = tid; d < head_dim; d += blockDim.x) {
            float pv = 0.f;
            for (int b = 0; b < tile_len; b++)
                pv += S_tile[b] * V_tile[b * (int)head_dim + (int)d];
            O_acc[d] = O_acc[d] * alpha + pv;
        }
        __syncthreads();

        m = m_new;
        l = l_new;
    }

    const float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
    for (size_t d = tid; d < head_dim; d += blockDim.x) {
        const float out = O_acc[d] * inv_l;
        store_f(attn_val, qi * n_heads * head_dim + h * head_dim + d, out);
    }
}

template <typename T, int BR>
static void launch_fa2(T *attn_val, const T *q, const T *k, const T *v, float scale, size_t qlen, size_t kvlen,
                       size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    const int blocks = (int)(qlen * n_heads);
    const int threads = 256;
    const size_t smem_bytes =
        (size_t)BR * head_dim * sizeof(float) * 2 + (size_t)BR * sizeof(float) + head_dim * sizeof(float) * 2;

    self_attn_fa2_kernel<T, BR><<<blocks, threads, smem_bytes>>>(
        attn_val, q, k, v, scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
}

namespace llaisys::ops::nvidia {

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads,
                    size_t head_dim) {
    const size_t smem_br32 =
        32u * head_dim * sizeof(float) * 2 + 32u * sizeof(float) + head_dim * sizeof(float) * 2;
    const size_t smem_br16 =
        16u * head_dim * sizeof(float) * 2 + 16u * sizeof(float) + head_dim * sizeof(float) * 2;

    auto pick_and_launch = [&]() {
        if (smem_br32 <= 48u * 1024u) {
            switch (type) {
            case LLAISYS_DTYPE_F32:
                launch_fa2<float, 32>((float *)attn_val, (const float *)q, (const float *)k, (const float *)v, scale,
                                      qlen, kvlen, n_heads, n_kv_heads, head_dim);
                return;
            case LLAISYS_DTYPE_BF16:
                launch_fa2<__nv_bfloat16, 32>((__nv_bfloat16 *)attn_val, (const __nv_bfloat16 *)q,
                                                (const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v, scale, qlen, kvlen,
                                                n_heads, n_kv_heads, head_dim);
                return;
            case LLAISYS_DTYPE_F16:
                launch_fa2<__half, 32>((__half *)attn_val, (const __half *)q, (const __half *)k, (const __half *)v,
                                       scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
                return;
            default: break;
            }
        }
        if (smem_br16 <= 96u * 1024u) {
            switch (type) {
            case LLAISYS_DTYPE_F32:
                launch_fa2<float, 16>((float *)attn_val, (const float *)q, (const float *)k, (const float *)v, scale,
                                      qlen, kvlen, n_heads, n_kv_heads, head_dim);
                return;
            case LLAISYS_DTYPE_BF16:
                launch_fa2<__nv_bfloat16, 16>((__nv_bfloat16 *)attn_val, (const __nv_bfloat16 *)q,
                                               (const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v, scale, qlen, kvlen,
                                               n_heads, n_kv_heads, head_dim);
                return;
            case LLAISYS_DTYPE_F16:
                launch_fa2<__half, 16>((__half *)attn_val, (const __half *)q, (const __half *)k, (const __half *)v,
                                       scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
                return;
            default: break;
            }
        }
        throw std::runtime_error("self_attention: head_dim too large for FA2-tile shared memory budget");
    };

    pick_and_launch();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("self_attention FA2 kernel: ") + cudaGetErrorString(err));
}

} // namespace llaisys::ops::nvidia
