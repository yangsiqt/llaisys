// self_attention_nvidia.cu — FlashAttention-style tiled attention (online softmax over KV blocks).
// One block per (query position, head). KV sequence is processed in tiles to bound shared memory
// and improve HBM reuse vs materializing full [kvlen] score rows.
#include "self_attention_nvidia.hpp"
#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

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
    val = (threadIdx.x < (unsigned)nw) ? warp_max[threadIdx.x] : -FLT_MAX;
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
    val = (threadIdx.x < (unsigned)nw) ? warp_sum[threadIdx.x] : 0.f;
    if (wid == 0) val = warp_reduce_sum(val);
    if (threadIdx.x == 0) warp_sum[0] = val;
    __syncthreads();
    return warp_sum[0];
}

// BR = max keys per tile (static upper bound; actual tile may be shorter).
// qlen_per_seq: number of query positions per sequence (1 for decode, seq_len for prefill).
// num_seqs = qlen / qlen_per_seq (batch size).
// K/V are stored as [num_seqs * kvlen_per_seq, n_kv_heads, head_dim] interleaved by sequence.
template <typename T, int BR>
__global__ void self_attn_fa2_kernel(T *attn_val, const T *q, const T *k, const T *v, float scale, size_t qlen,
                                     size_t kvlen, size_t n_heads, size_t n_kv_heads, size_t head_dim,
                                     size_t qlen_per_seq) {
    const size_t qi = blockIdx.x / n_heads;
    const size_t h  = blockIdx.x % n_heads;
    const size_t seq_id = qi / qlen_per_seq;
    const size_t local_qi = qi % qlen_per_seq;
    const size_t num_seqs = qlen / qlen_per_seq;
    const size_t kvlen_per_seq = kvlen / num_seqs;
    const size_t kv_h = h / (n_heads / n_kv_heads);
    const unsigned tid = threadIdx.x;
    // causal mask: within this sequence, forward positions are masked
    const size_t abs_local_qi = kvlen_per_seq - qlen_per_seq + local_qi;

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
            const size_t ki_seq_id = ki / kvlen_per_seq;
            const size_t local_ki = ki % kvlen_per_seq;
            if (ki_seq_id != seq_id) sc = -FLT_MAX;
            else if (local_ki > abs_local_qi) sc = -FLT_MAX;
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
                       size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t qlen_per_seq) {
    const int blocks = (int)(qlen * n_heads);
    const int threads = 256;
    const size_t smem_bytes =
        (size_t)BR * head_dim * sizeof(float) * 2 + (size_t)BR * sizeof(float) + head_dim * sizeof(float) * 2;

    self_attn_fa2_kernel<T, BR><<<blocks, threads, smem_bytes>>>(
        attn_val, q, k, v, scale, qlen, kvlen, n_heads, n_kv_heads, head_dim, qlen_per_seq);
}

template <typename T, int BR>
__global__ void self_attn_slots_decode_kernel(T *attn_val, const T *q, const T *k_cache, const T *v_cache,
                                              const int64_t *slot_ids, const int64_t *seq_lens, float scale,
                                              size_t batch_size, size_t maxseq, size_t n_heads,
                                              size_t n_kv_heads, size_t head_dim) {
    const size_t bi = blockIdx.x / n_heads;
    const size_t h = blockIdx.x % n_heads;
    if (bi >= batch_size) return;

    const int64_t slot_i64 = slot_ids[bi];
    const int64_t seq_i64 = seq_lens[bi];
    if (slot_i64 < 0 || seq_i64 < 0) return;

    const size_t slot_id = static_cast<size_t>(slot_i64);
    const size_t kv_len = static_cast<size_t>(seq_i64) + 1;
    const size_t kv_h = h / (n_heads / n_kv_heads);
    const unsigned tid = threadIdx.x;

    extern __shared__ unsigned char smem_u[];
    float *const K_tile = reinterpret_cast<float *>(smem_u);
    float *const V_tile = K_tile + BR * head_dim;
    float *const S_tile = V_tile + BR * head_dim;
    float *const Q_tile = S_tile + BR;
    float *const O_acc = Q_tile + head_dim;

    for (size_t d = tid; d < head_dim; d += blockDim.x)
        Q_tile[d] = load_f(q, bi * n_heads * head_dim + h * head_dim + d);
    __syncthreads();

    float m = -FLT_MAX;
    float l = 0.f;
    for (size_t d = tid; d < head_dim; d += blockDim.x) O_acc[d] = 0.f;
    __syncthreads();

    for (int k0 = 0; k0 < (int)kv_len; k0 += BR) {
        const int tile_len = min(BR, (int)kv_len - k0);

        for (int idx = tid; idx < tile_len * (int)head_dim; idx += blockDim.x) {
            const int b = idx / (int)head_dim;
            const int d = idx % (int)head_dim;
            const size_t ki = (size_t)k0 + (size_t)b;
            const size_t off = (((slot_id * maxseq + ki) * n_kv_heads + kv_h) * head_dim + (size_t)d);
            K_tile[b * (int)head_dim + d] = load_f(k_cache, off);
            V_tile[b * (int)head_dim + d] = load_f(v_cache, off);
        }
        __syncthreads();

        for (int b = tid; b < tile_len; b += blockDim.x) {
            float sc = 0.f;
            for (size_t d = 0; d < head_dim; d++)
                sc += Q_tile[d] * K_tile[b * (int)head_dim + (int)d];
            S_tile[b] = sc * scale;
        }
        __syncthreads();

        float local_max = -FLT_MAX;
        for (int b = tid; b < tile_len; b += blockDim.x) local_max = fmaxf(local_max, S_tile[b]);
        const float m_tile = block_reduce_max(local_max);
        const float m_new = fmaxf(m, m_tile);
        __syncthreads();

        for (int b = tid; b < tile_len; b += blockDim.x) {
            const float pb = expf(S_tile[b] - m_new);
            S_tile[b] = pb;
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
        store_f(attn_val, bi * n_heads * head_dim + h * head_dim + d, out);
    }
}

template <typename T, int BR>
static void launch_slots_decode(T *attn_val, const T *q, const T *k_cache, const T *v_cache,
                                const int64_t *slot_ids, const int64_t *seq_lens, float scale,
                                size_t batch_size, size_t maxseq, size_t n_heads, size_t n_kv_heads,
                                size_t head_dim) {
    const int blocks = (int)(batch_size * n_heads);
    const int threads = 256;
    const size_t smem_bytes =
        (size_t)BR * head_dim * sizeof(float) * 2 + (size_t)BR * sizeof(float) + head_dim * sizeof(float) * 2;

    if (smem_bytes > 48u * 1024u) {
        cudaFuncSetAttribute(self_attn_slots_decode_kernel<T, BR>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
    }
    self_attn_slots_decode_kernel<T, BR><<<blocks, threads, smem_bytes>>>(
        attn_val, q, k_cache, v_cache, slot_ids, seq_lens, scale, batch_size, maxseq, n_heads, n_kv_heads,
        head_dim);
}

template <typename T, int BR>
__global__ void self_attn_paged_slots_decode_kernel(T *attn_val, const T *q, const T *k_cache,
                                                    const T *v_cache, const int64_t *block_tables,
                                                    const int64_t *slot_ids, const int64_t *seq_lens,
                                                    float scale, size_t batch_size,
                                                    size_t max_blocks_per_slot, size_t block_size,
                                                    size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    const size_t bi = blockIdx.x / n_heads;
    const size_t h = blockIdx.x % n_heads;
    if (bi >= batch_size) return;

    const int64_t slot_i64 = slot_ids[bi];
    const int64_t seq_i64 = seq_lens[bi];
    if (slot_i64 < 0 || seq_i64 < 0) return;

    const size_t slot_id = static_cast<size_t>(slot_i64);
    const size_t kv_len = static_cast<size_t>(seq_i64) + 1;
    const size_t kv_h = h / (n_heads / n_kv_heads);
    const unsigned tid = threadIdx.x;

    extern __shared__ unsigned char smem_u[];
    float *const K_tile = reinterpret_cast<float *>(smem_u);
    float *const V_tile = K_tile + BR * head_dim;
    float *const S_tile = V_tile + BR * head_dim;
    float *const Q_tile = S_tile + BR;
    float *const O_acc = Q_tile + head_dim;

    for (size_t d = tid; d < head_dim; d += blockDim.x)
        Q_tile[d] = load_f(q, bi * n_heads * head_dim + h * head_dim + d);
    __syncthreads();

    float m = -FLT_MAX;
    float l = 0.f;
    for (size_t d = tid; d < head_dim; d += blockDim.x) O_acc[d] = 0.f;
    __syncthreads();

    for (int k0 = 0; k0 < (int)kv_len; k0 += BR) {
        const int tile_len = min(BR, (int)kv_len - k0);
        for (int idx = tid; idx < tile_len * (int)head_dim; idx += blockDim.x) {
            const int b = idx / (int)head_dim;
            const int d = idx % (int)head_dim;
            const size_t ki = (size_t)k0 + (size_t)b;
            const size_t logical_block = ki / block_size;
            const size_t block_offset = ki % block_size;
            const int64_t physical_block = block_tables[slot_id * max_blocks_per_slot + logical_block];
            float kval = 0.f, vval = 0.f;
            if (physical_block >= 0) {
                const size_t off = (((size_t)physical_block * block_size + block_offset) * n_kv_heads + kv_h) *
                                       head_dim + (size_t)d;
                kval = load_f(k_cache, off);
                vval = load_f(v_cache, off);
            }
            K_tile[b * (int)head_dim + d] = kval;
            V_tile[b * (int)head_dim + d] = vval;
        }
        __syncthreads();

        for (int b = tid; b < tile_len; b += blockDim.x) {
            float sc = 0.f;
            for (size_t d = 0; d < head_dim; d++)
                sc += Q_tile[d] * K_tile[b * (int)head_dim + (int)d];
            S_tile[b] = sc * scale;
        }
        __syncthreads();

        float local_max = -FLT_MAX;
        for (int b = tid; b < tile_len; b += blockDim.x) local_max = fmaxf(local_max, S_tile[b]);
        const float m_tile = block_reduce_max(local_max);
        const float m_new = fmaxf(m, m_tile);
        __syncthreads();

        for (int b = tid; b < tile_len; b += blockDim.x) S_tile[b] = expf(S_tile[b] - m_new);
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
        store_f(attn_val, bi * n_heads * head_dim + h * head_dim + d, O_acc[d] * inv_l);
    }
}

template <typename T, int BR>
static void launch_paged_slots_decode(T *attn_val, const T *q, const T *k_cache, const T *v_cache,
                                      const int64_t *block_tables, const int64_t *slot_ids,
                                      const int64_t *seq_lens, float scale, size_t batch_size,
                                      size_t max_blocks_per_slot, size_t block_size, size_t n_heads,
                                      size_t n_kv_heads, size_t head_dim) {
    const int blocks = (int)(batch_size * n_heads);
    const int threads = 256;
    const size_t smem_bytes =
        (size_t)BR * head_dim * sizeof(float) * 2 + (size_t)BR * sizeof(float) + head_dim * sizeof(float) * 2;
    if (smem_bytes > 48u * 1024u) {
        cudaFuncSetAttribute(self_attn_paged_slots_decode_kernel<T, BR>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }
    self_attn_paged_slots_decode_kernel<T, BR><<<blocks, threads, smem_bytes>>>(
        attn_val, q, k_cache, v_cache, block_tables, slot_ids, seq_lens, scale, batch_size,
        max_blocks_per_slot, block_size, n_heads, n_kv_heads, head_dim);
}

template <typename T, int BR, int MAX_GROUP>
__global__ void self_attn_gqa_slots_decode_kernel(T *attn_val, const T *q, const T *k_cache, const T *v_cache,
                                                  const int64_t *slot_ids, const int64_t *seq_lens, float scale,
                                                  size_t batch_size, size_t maxseq, size_t n_heads,
                                                  size_t n_kv_heads, size_t head_dim) {
    const size_t bi = blockIdx.x / n_kv_heads;
    const size_t kv_h = blockIdx.x % n_kv_heads;
    if (bi >= batch_size) return;

    const int64_t slot_i64 = slot_ids[bi];
    const int64_t seq_i64 = seq_lens[bi];
    if (slot_i64 < 0 || seq_i64 < 0) return;

    const size_t slot_id = static_cast<size_t>(slot_i64);
    const size_t kv_len = static_cast<size_t>(seq_i64) + 1;
    const size_t group = n_heads / n_kv_heads;
    if (group > MAX_GROUP) return;
    const unsigned tid = threadIdx.x;

    extern __shared__ unsigned char smem_u[];
    float *const K_tile = reinterpret_cast<float *>(smem_u);
    float *const V_tile = K_tile + BR * head_dim;
    float *const Q_tile = V_tile + BR * head_dim;
    float *const O_acc = Q_tile + MAX_GROUP * head_dim;
    float *const S_tile = O_acc + MAX_GROUP * head_dim;

    for (int gh = 0; gh < (int)group; gh++) {
        const size_t h = kv_h * group + (size_t)gh;
        for (size_t d = tid; d < head_dim; d += blockDim.x) {
            Q_tile[gh * (int)head_dim + d] =
                load_f(q, bi * n_heads * head_dim + h * head_dim + d);
            O_acc[gh * (int)head_dim + d] = 0.f;
        }
    }
    __syncthreads();

    float m[MAX_GROUP];
    float l[MAX_GROUP];
    for (int gh = 0; gh < MAX_GROUP; gh++) {
        m[gh] = -FLT_MAX;
        l[gh] = 0.f;
    }

    for (int k0 = 0; k0 < (int)kv_len; k0 += BR) {
        const int tile_len = min(BR, (int)kv_len - k0);

        for (int idx = tid; idx < tile_len * (int)head_dim; idx += blockDim.x) {
            const int b = idx / (int)head_dim;
            const int d = idx % (int)head_dim;
            const size_t ki = (size_t)k0 + (size_t)b;
            const size_t off = (((slot_id * maxseq + ki) * n_kv_heads + kv_h) * head_dim + (size_t)d);
            K_tile[b * (int)head_dim + d] = load_f(k_cache, off);
            V_tile[b * (int)head_dim + d] = load_f(v_cache, off);
        }
        __syncthreads();

        for (int gh = 0; gh < (int)group; gh++) {
            float *S = S_tile + gh * BR;
            for (int b = tid; b < tile_len; b += blockDim.x) {
                float sc = 0.f;
                for (size_t d = 0; d < head_dim; d++) {
                    sc += Q_tile[gh * (int)head_dim + (int)d] *
                          K_tile[b * (int)head_dim + (int)d];
                }
                S[b] = sc * scale;
            }
            __syncthreads();

            float local_max = -FLT_MAX;
            for (int b = tid; b < tile_len; b += blockDim.x) local_max = fmaxf(local_max, S[b]);
            const float m_tile = block_reduce_max(local_max);
            const float m_new = fmaxf(m[gh], m_tile);
            __syncthreads();

            for (int b = tid; b < tile_len; b += blockDim.x) {
                const float pb = expf(S[b] - m_new);
                S[b] = pb;
            }
            __syncthreads();

            float local_sum = 0.f;
            for (int b = tid; b < tile_len; b += blockDim.x) local_sum += S[b];
            const float sum_p = block_reduce_sum(local_sum);

            const float alpha = expf(m[gh] - m_new);
            const float l_new = alpha * l[gh] + sum_p;

            for (size_t d = tid; d < head_dim; d += blockDim.x) {
                float pv = 0.f;
                for (int b = 0; b < tile_len; b++)
                    pv += S[b] * V_tile[b * (int)head_dim + (int)d];
                O_acc[gh * (int)head_dim + (int)d] =
                    O_acc[gh * (int)head_dim + (int)d] * alpha + pv;
            }
            __syncthreads();

            m[gh] = m_new;
            l[gh] = l_new;
        }
    }

    for (int gh = 0; gh < (int)group; gh++) {
        const float inv_l = (l[gh] > 0.f) ? (1.f / l[gh]) : 0.f;
        const size_t h = kv_h * group + (size_t)gh;
        for (size_t d = tid; d < head_dim; d += blockDim.x) {
            const float out = O_acc[gh * (int)head_dim + (int)d] * inv_l;
            store_f(attn_val, bi * n_heads * head_dim + h * head_dim + d, out);
        }
    }
}

template <typename T, int BR>
static void launch_gqa_slots_decode(T *attn_val, const T *q, const T *k_cache, const T *v_cache,
                                    const int64_t *slot_ids, const int64_t *seq_lens, float scale,
                                    size_t batch_size, size_t maxseq, size_t n_heads, size_t n_kv_heads,
                                    size_t head_dim) {
    constexpr int MAX_GROUP = 8;
    const int blocks = (int)(batch_size * n_kv_heads);
    const int threads = 256;
    const size_t smem_bytes =
        (size_t)BR * head_dim * sizeof(float) * 2 +
        (size_t)MAX_GROUP * head_dim * sizeof(float) * 2 +
        (size_t)MAX_GROUP * BR * sizeof(float);
    if (smem_bytes > 48u * 1024u) {
        cudaFuncSetAttribute(self_attn_gqa_slots_decode_kernel<T, BR, MAX_GROUP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
    }
    self_attn_gqa_slots_decode_kernel<T, BR, MAX_GROUP><<<blocks, threads, smem_bytes>>>(
        attn_val, q, k_cache, v_cache, slot_ids, seq_lens, scale, batch_size, maxseq, n_heads, n_kv_heads,
        head_dim);
}

namespace llaisys::ops::nvidia {

// qlen_per_seq: number of query tokens per sequence (1 for decode, seq_len for prefill).
// For bs=1 legacy path: qlen_per_seq = qlen.
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads,
                    size_t head_dim, size_t qlen_per_seq) {
    const size_t smem_br32 =
        32u * head_dim * sizeof(float) * 2 + 32u * sizeof(float) + head_dim * sizeof(float) * 2;
    const size_t smem_br16 =
        16u * head_dim * sizeof(float) * 2 + 16u * sizeof(float) + head_dim * sizeof(float) * 2;

    auto pick_and_launch = [&]() {
        if (smem_br32 <= 48u * 1024u) {
            switch (type) {
            case LLAISYS_DTYPE_F32:
                launch_fa2<float, 32>((float *)attn_val, (const float *)q, (const float *)k, (const float *)v, scale,
                                      qlen, kvlen, n_heads, n_kv_heads, head_dim, qlen_per_seq);
                return;
            case LLAISYS_DTYPE_BF16:
                launch_fa2<__nv_bfloat16, 32>((__nv_bfloat16 *)attn_val, (const __nv_bfloat16 *)q,
                                                (const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v, scale, qlen, kvlen,
                                                n_heads, n_kv_heads, head_dim, qlen_per_seq);
                return;
            case LLAISYS_DTYPE_F16:
                launch_fa2<__half, 32>((__half *)attn_val, (const __half *)q, (const __half *)k, (const __half *)v,
                                       scale, qlen, kvlen, n_heads, n_kv_heads, head_dim, qlen_per_seq);
                return;
            default: break;
            }
        }
        if (smem_br16 <= 96u * 1024u) {
            switch (type) {
            case LLAISYS_DTYPE_F32:
                launch_fa2<float, 16>((float *)attn_val, (const float *)q, (const float *)k, (const float *)v, scale,
                                      qlen, kvlen, n_heads, n_kv_heads, head_dim, qlen_per_seq);
                return;
            case LLAISYS_DTYPE_BF16:
                launch_fa2<__nv_bfloat16, 16>((__nv_bfloat16 *)attn_val, (const __nv_bfloat16 *)q,
                                               (const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v, scale, qlen, kvlen,
                                               n_heads, n_kv_heads, head_dim, qlen_per_seq);
                return;
            case LLAISYS_DTYPE_F16:
                launch_fa2<__half, 16>((__half *)attn_val, (const __half *)q, (const __half *)k, (const __half *)v,
                                       scale, qlen, kvlen, n_heads, n_kv_heads, head_dim, qlen_per_seq);
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

void self_attention_slots_decode(std::byte *attn_val, const std::byte *q, const std::byte *k_cache,
                                 const std::byte *v_cache, const int64_t *slot_ids, const int64_t *seq_lens,
                                 float scale, llaisysDataType_t type, size_t batch_size, size_t maxseq,
                                 size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    const size_t smem_br64 =
        64u * head_dim * sizeof(float) * 2 + 64u * sizeof(float) + head_dim * sizeof(float) * 2;
    const size_t smem_br32 =
        32u * head_dim * sizeof(float) * 2 + 32u * sizeof(float) + head_dim * sizeof(float) * 2;
    const size_t smem_br16 =
        16u * head_dim * sizeof(float) * 2 + 16u * sizeof(float) + head_dim * sizeof(float) * 2;

    auto pick_and_launch = [&]() {
        const char *force_br = std::getenv("LLAISYS_SLOT_ATTN_BR");
        const int forced_br = force_br ? std::atoi(force_br) : 0;
        const bool prefer_br64 = forced_br == 64 || (forced_br == 0 && head_dim == 128);
        if (prefer_br64 && smem_br64 <= 96u * 1024u) {
            switch (type) {
            case LLAISYS_DTYPE_F32:
                launch_slots_decode<float, 64>((float *)attn_val, (const float *)q, (const float *)k_cache,
                                               (const float *)v_cache, slot_ids, seq_lens, scale, batch_size, maxseq,
                                               n_heads, n_kv_heads, head_dim);
                return;
            case LLAISYS_DTYPE_BF16:
                launch_slots_decode<__nv_bfloat16, 64>((__nv_bfloat16 *)attn_val, (const __nv_bfloat16 *)q,
                                                       (const __nv_bfloat16 *)k_cache,
                                                       (const __nv_bfloat16 *)v_cache, slot_ids, seq_lens, scale,
                                                       batch_size, maxseq, n_heads, n_kv_heads, head_dim);
                return;
            case LLAISYS_DTYPE_F16:
                launch_slots_decode<__half, 64>((__half *)attn_val, (const __half *)q,
                                                (const __half *)k_cache, (const __half *)v_cache, slot_ids, seq_lens,
                                                scale, batch_size, maxseq, n_heads, n_kv_heads, head_dim);
                return;
            default: break;
            }
        }
        if ((forced_br == 0 || forced_br == 32) && smem_br32 <= 48u * 1024u) {
            switch (type) {
            case LLAISYS_DTYPE_F32:
                launch_slots_decode<float, 32>((float *)attn_val, (const float *)q, (const float *)k_cache,
                                               (const float *)v_cache, slot_ids, seq_lens, scale, batch_size, maxseq,
                                               n_heads, n_kv_heads, head_dim);
                return;
            case LLAISYS_DTYPE_BF16:
                launch_slots_decode<__nv_bfloat16, 32>((__nv_bfloat16 *)attn_val, (const __nv_bfloat16 *)q,
                                                       (const __nv_bfloat16 *)k_cache,
                                                       (const __nv_bfloat16 *)v_cache, slot_ids, seq_lens, scale,
                                                       batch_size, maxseq, n_heads, n_kv_heads, head_dim);
                return;
            case LLAISYS_DTYPE_F16:
                launch_slots_decode<__half, 32>((__half *)attn_val, (const __half *)q, (const __half *)k_cache,
                                                (const __half *)v_cache, slot_ids, seq_lens, scale, batch_size,
                                                maxseq, n_heads, n_kv_heads, head_dim);
                return;
            default: break;
            }
        }
        if ((forced_br == 0 || forced_br == 16) && smem_br16 <= 96u * 1024u) {
            switch (type) {
            case LLAISYS_DTYPE_F32:
                launch_slots_decode<float, 16>((float *)attn_val, (const float *)q, (const float *)k_cache,
                                               (const float *)v_cache, slot_ids, seq_lens, scale, batch_size, maxseq,
                                               n_heads, n_kv_heads, head_dim);
                return;
            case LLAISYS_DTYPE_BF16:
                launch_slots_decode<__nv_bfloat16, 16>((__nv_bfloat16 *)attn_val, (const __nv_bfloat16 *)q,
                                                       (const __nv_bfloat16 *)k_cache,
                                                       (const __nv_bfloat16 *)v_cache, slot_ids, seq_lens, scale,
                                                       batch_size, maxseq, n_heads, n_kv_heads, head_dim);
                return;
            case LLAISYS_DTYPE_F16:
                launch_slots_decode<__half, 16>((__half *)attn_val, (const __half *)q, (const __half *)k_cache,
                                                (const __half *)v_cache, slot_ids, seq_lens, scale, batch_size,
                                                maxseq, n_heads, n_kv_heads, head_dim);
                return;
            default: break;
            }
        }
        throw std::runtime_error("self_attention_slots_decode: head_dim too large for shared memory budget");
    };

    pick_and_launch();
}

void self_attention_gqa_slots_decode(std::byte *attn_val, const std::byte *q, const std::byte *k_cache,
                                     const std::byte *v_cache, const int64_t *slot_ids, const int64_t *seq_lens,
                                     float scale, llaisysDataType_t type, size_t batch_size, size_t maxseq,
                                     size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    const size_t group = n_heads / n_kv_heads;
    if (head_dim != 128 || group == 0 || group > 8) {
        return self_attention_slots_decode(attn_val, q, k_cache, v_cache, slot_ids, seq_lens, scale, type,
                                           batch_size, maxseq, n_heads, n_kv_heads, head_dim);
    }

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_gqa_slots_decode<float, 32>((float *)attn_val, (const float *)q, (const float *)k_cache,
                                                 (const float *)v_cache, slot_ids, seq_lens, scale, batch_size,
                                                 maxseq, n_heads, n_kv_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return launch_gqa_slots_decode<__nv_bfloat16, 32>(
            (__nv_bfloat16 *)attn_val, (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k_cache,
            (const __nv_bfloat16 *)v_cache, slot_ids, seq_lens, scale, batch_size, maxseq, n_heads, n_kv_heads,
            head_dim);
    case LLAISYS_DTYPE_F16:
        return launch_gqa_slots_decode<__half, 32>((__half *)attn_val, (const __half *)q,
                                                  (const __half *)k_cache, (const __half *)v_cache, slot_ids,
                                                  seq_lens, scale, batch_size, maxseq, n_heads, n_kv_heads,
                                                  head_dim);
    default:
        throw std::runtime_error("Unsupported dtype for self_attention_gqa_slots_decode");
    }
}

void self_attention_paged_slots_decode(std::byte *attn_val, const std::byte *q,
                                       const std::byte *k_cache, const std::byte *v_cache,
                                       const int64_t *block_tables, const int64_t *slot_ids,
                                       const int64_t *seq_lens, float scale, llaisysDataType_t type,
                                       size_t batch_size, size_t max_blocks_per_slot,
                                       size_t block_size, size_t n_heads, size_t n_kv_heads,
                                       size_t head_dim) {
    const size_t smem_br64 =
        64u * head_dim * sizeof(float) * 2 + 64u * sizeof(float) + head_dim * sizeof(float) * 2;
    const size_t smem_br32 =
        32u * head_dim * sizeof(float) * 2 + 32u * sizeof(float) + head_dim * sizeof(float) * 2;
    const bool use_br64 = head_dim == 128 && smem_br64 <= 96u * 1024u;
    if (use_br64) {
        switch (type) {
        case LLAISYS_DTYPE_F32:
            return launch_paged_slots_decode<float, 64>((float *)attn_val, (const float *)q,
                                                        (const float *)k_cache, (const float *)v_cache,
                                                        block_tables, slot_ids, seq_lens, scale, batch_size,
                                                        max_blocks_per_slot, block_size, n_heads, n_kv_heads,
                                                        head_dim);
        case LLAISYS_DTYPE_BF16:
            return launch_paged_slots_decode<__nv_bfloat16, 64>(
                (__nv_bfloat16 *)attn_val, (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k_cache,
                (const __nv_bfloat16 *)v_cache, block_tables, slot_ids, seq_lens, scale, batch_size,
                max_blocks_per_slot, block_size, n_heads, n_kv_heads, head_dim);
        case LLAISYS_DTYPE_F16:
            return launch_paged_slots_decode<__half, 64>((__half *)attn_val, (const __half *)q,
                                                        (const __half *)k_cache, (const __half *)v_cache,
                                                        block_tables, slot_ids, seq_lens, scale, batch_size,
                                                        max_blocks_per_slot, block_size, n_heads, n_kv_heads,
                                                        head_dim);
        default: break;
        }
    }
    if (smem_br32 <= 48u * 1024u) {
        switch (type) {
        case LLAISYS_DTYPE_F32:
            return launch_paged_slots_decode<float, 32>((float *)attn_val, (const float *)q,
                                                        (const float *)k_cache, (const float *)v_cache,
                                                        block_tables, slot_ids, seq_lens, scale, batch_size,
                                                        max_blocks_per_slot, block_size, n_heads, n_kv_heads,
                                                        head_dim);
        case LLAISYS_DTYPE_BF16:
            return launch_paged_slots_decode<__nv_bfloat16, 32>(
                (__nv_bfloat16 *)attn_val, (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k_cache,
                (const __nv_bfloat16 *)v_cache, block_tables, slot_ids, seq_lens, scale, batch_size,
                max_blocks_per_slot, block_size, n_heads, n_kv_heads, head_dim);
        case LLAISYS_DTYPE_F16:
            return launch_paged_slots_decode<__half, 32>((__half *)attn_val, (const __half *)q,
                                                        (const __half *)k_cache, (const __half *)v_cache,
                                                        block_tables, slot_ids, seq_lens, scale, batch_size,
                                                        max_blocks_per_slot, block_size, n_heads, n_kv_heads,
                                                        head_dim);
        default: break;
        }
    }
    throw std::runtime_error("Unsupported dtype/shape for self_attention_paged_slots_decode");
}

} // namespace llaisys::ops::nvidia
