// self_attention_nvidia.cu — FA2-style tiled attention + online softmax (bounded shared memory)
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
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    return v;
}
__device__ inline float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ float block_reduce_max(float v, float *warp_buf) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nw = (blockDim.x + 31) >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) warp_buf[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < nw) ? warp_buf[lane] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (lane == 0) warp_buf[0] = v;
    }
    __syncthreads();
    return warp_buf[0];
}
__device__ float block_reduce_sum(float v, float *warp_buf) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nw = (blockDim.x + 31) >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) warp_buf[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < nw) ? warp_buf[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) warp_buf[0] = v;
    }
    __syncthreads();
    return warp_buf[0];
}

// smem: s_tile[BK], o_accum[head_dim], warp_tmp[32]
template <typename T, int BK>
__global__ void self_attn_fa2_kernel(T *attn_val, const T *q, const T *k, const T *v, float scale,
                                     size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads,
                                     size_t head_dim) {
    const size_t qi = blockIdx.x / n_heads;
    const size_t h = blockIdx.x % n_heads;
    const size_t kv_h = h / (n_heads / n_kv_heads);
    const size_t tid = threadIdx.x;
    const size_t abs_qi = kvlen - qlen + qi;

    extern __shared__ float smem_raw[];
    float *const s_tile = smem_raw;
    float *const o_accum = smem_raw + BK;
    float *const warp_tmp = smem_raw + BK + head_dim;

    const size_t q_base = qi * n_heads * head_dim + h * head_dim;

    __shared__ float s_m, s_l;
    if (tid == 0) {
        s_m = -FLT_MAX;
        s_l = 0.f;
    }
    __syncthreads();

    for (size_t d = tid; d < head_dim; d += blockDim.x) o_accum[d] = 0.f;
    __syncthreads();

    for (size_t tile0 = 0; tile0 < kvlen; tile0 += BK) {
        const size_t tile_len = min((size_t)BK, kvlen - tile0);
        __syncthreads();
        const float m_old = s_m;

        for (size_t tk = tid; tk < tile_len; tk += blockDim.x) {
            const size_t ki = tile0 + tk;
            float sc = 0.f;
            for (size_t d = 0; d < head_dim; d++)
                sc += load_f(q, q_base + d) *
                      load_f(k, ki * n_kv_heads * head_dim + kv_h * head_dim + d);
            s_tile[tk] = (ki <= abs_qi) ? sc * scale : -FLT_MAX;
        }
        __syncthreads();

        float tmax = -FLT_MAX;
        for (size_t tk = tid; tk < tile_len; tk += blockDim.x) tmax = fmaxf(tmax, s_tile[tk]);
        tmax = block_reduce_max(tmax, warp_tmp);

        const float m_new = fmaxf(m_old, tmax);
        const float alpha = expf(m_old - m_new);

        float psum = 0.f;
        for (size_t tk = tid; tk < tile_len; tk += blockDim.x) psum += expf(s_tile[tk] - m_new);
        psum = block_reduce_sum(psum, warp_tmp);

        if (tid == 0) {
            s_l = s_l * alpha + psum;
            s_m = m_new;
        }
        __syncthreads();

        for (size_t d = tid; d < head_dim; d += blockDim.x) {
            float sum_wv = 0.f;
            for (size_t tk = 0; tk < tile_len; tk++)
                sum_wv += expf(s_tile[tk] - m_new) *
                          load_f(v, (tile0 + tk) * n_kv_heads * head_dim + kv_h * head_dim + d);
            o_accum[d] = o_accum[d] * alpha + sum_wv;
        }
        __syncthreads();
    }

    const float inv_l = 1.f / fmaxf(s_l, 1e-20f);
    for (size_t d = tid; d < head_dim; d += blockDim.x)
        store_f(attn_val, q_base + d, o_accum[d] * inv_l);
}

namespace llaisys::ops::nvidia {

namespace {
constexpr int kKvTile = 128;

size_t fa2_smem_bytes(size_t head_dim) {
    return (kKvTile + head_dim) * sizeof(float) + 32 * sizeof(float);
}
} // namespace

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t type, size_t qlen, size_t kvlen, size_t n_heads,
                    size_t n_kv_heads, size_t head_dim) {
    const int blocks = (int)(qlen * n_heads);
    const int threads = 256;
    const size_t smem = fa2_smem_bytes(head_dim);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attn_fa2_kernel<float, kKvTile><<<blocks, threads, smem>>>(
            (float *)attn_val, (const float *)q, (const float *)k, (const float *)v, scale, qlen,
            kvlen, n_heads, n_kv_heads, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attn_fa2_kernel<__nv_bfloat16, kKvTile><<<blocks, threads, smem>>>(
            (__nv_bfloat16 *)attn_val, (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k,
            (const __nv_bfloat16 *)v, scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        self_attn_fa2_kernel<__half, kKvTile><<<blocks, threads, smem>>>(
            (__half *)attn_val, (const __half *)q, (const __half *)k, (const __half *)v, scale, qlen,
            kvlen, n_heads, n_kv_heads, head_dim);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for CUDA self_attention");
    }
}
} // namespace llaisys::ops::nvidia
