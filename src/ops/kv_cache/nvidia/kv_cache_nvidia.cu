#include "kv_cache_nvidia.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <stdexcept>

template <typename T>
__device__ inline float kv_to_f(T v) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(v);
    else if constexpr (std::is_same_v<T, __half>) return __half2float(v);
    else return (float)v;
}

template <typename T>
__device__ inline T kv_from_f(float v) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) return __float2bfloat16(v);
    else if constexpr (std::is_same_v<T, __half>) return __float2half(v);
    else return (T)v;
}

template <typename T>
__global__ void scatter_kv_decode_kernel(T *k_cache, T *v_cache, const T *k, const T *v,
                                         const int64_t *slot_ids, const int64_t *positions,
                                         size_t maxseq, size_t n_kv_heads, size_t head_dim,
                                         size_t batch_size, size_t elems_per_batch) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * elems_per_batch;
    if (idx >= total) return;

    size_t b = idx / elems_per_batch;
    size_t rem = idx - b * elems_per_batch;
    size_t kv_h = rem / head_dim;
    size_t d = rem - kv_h * head_dim;

    int64_t slot_i64 = slot_ids[b];
    int64_t pos_i64 = positions[b];
    if (slot_i64 < 0 || pos_i64 < 0) return;

    size_t slot = (size_t)slot_i64;
    size_t pos = (size_t)pos_i64;
    size_t dst = (((slot * maxseq + pos) * n_kv_heads + kv_h) * head_dim + d);
    size_t src = b * elems_per_batch + rem;
    k_cache[dst] = k[src];
    v_cache[dst] = v[src];
}

template <typename T>
__global__ void rope_and_scatter_kv_decode_kernel(T *q, T *k, const T *v, T *k_cache, T *v_cache,
                                                  const int64_t *pos_ids, const int64_t *slot_ids,
                                                  const int64_t *positions, float theta,
                                                  size_t batch_size, size_t maxseq, size_t n_heads,
                                                  size_t n_kv_heads, size_t head_dim, size_t max_heads) {
    size_t half_dim = head_dim / 2;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * max_heads * half_dim;
    if (idx >= total) return;

    size_t j = idx % half_dim;
    size_t h = (idx / half_dim) % max_heads;
    size_t b = idx / (half_dim * max_heads);

    int64_t pos_i64 = pos_ids[b];
    int64_t slot_i64 = slot_ids[b];
    int64_t cache_pos_i64 = positions[b];
    if (pos_i64 < 0 || slot_i64 < 0 || cache_pos_i64 < 0) return;

    float freq = (float)pos_i64 / powf(theta, (2.f * (float)j) / (float)head_dim);
    float cos_f, sin_f;
    __sincosf(freq, &sin_f, &cos_f);

    if (h < n_heads) {
        size_t q_base = (b * n_heads + h) * head_dim;
        float a = kv_to_f(q[q_base + j]);
        float c = kv_to_f(q[q_base + j + half_dim]);
        q[q_base + j] = kv_from_f<T>(a * cos_f - c * sin_f);
        q[q_base + j + half_dim] = kv_from_f<T>(c * cos_f + a * sin_f);
    }

    if (h < n_kv_heads) {
        size_t kv_base = (b * n_kv_heads + h) * head_dim;
        float a = kv_to_f(k[kv_base + j]);
        float c = kv_to_f(k[kv_base + j + half_dim]);
        T rk0 = kv_from_f<T>(a * cos_f - c * sin_f);
        T rk1 = kv_from_f<T>(c * cos_f + a * sin_f);
        k[kv_base + j] = rk0;
        k[kv_base + j + half_dim] = rk1;

        size_t slot = (size_t)slot_i64;
        size_t cache_pos = (size_t)cache_pos_i64;
        size_t cache_base = ((slot * maxseq + cache_pos) * n_kv_heads + h) * head_dim;
        k_cache[cache_base + j] = rk0;
        k_cache[cache_base + j + half_dim] = rk1;
        v_cache[cache_base + j] = v[kv_base + j];
        v_cache[cache_base + j + half_dim] = v[kv_base + j + half_dim];
    }
}

template <typename T>
__global__ void split_qkv_decode_kernel(T *q, T *k, T *v, const T *qkv,
                                        size_t batch_size, size_t q_dim, size_t k_dim,
                                        size_t v_dim, size_t total_dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * total_dim;
    if (idx >= total) return;
    size_t b = idx / total_dim;
    size_t col = idx - b * total_dim;
    T val = qkv[idx];
    if (col < q_dim) {
        q[b * q_dim + col] = val;
    } else if (col < q_dim + k_dim) {
        size_t k_col = col - q_dim;
        k[b * k_dim + k_col] = val;
    } else {
        size_t v_col = col - q_dim - k_dim;
        v[b * v_dim + v_col] = val;
    }
}

template <typename T>
__global__ void split_gate_up_decode_kernel(T *gate, T *up, const T *gate_up,
                                            size_t batch_size, size_t dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * dim * 2;
    if (idx >= total) return;
    size_t b = idx / (dim * 2);
    size_t col = idx - b * dim * 2;
    T val = gate_up[idx];
    if (col < dim) {
        gate[b * dim + col] = val;
    } else {
        up[b * dim + (col - dim)] = val;
    }
}

template <typename T>
__global__ void copy_kv_slots_to_blocks_kernel(T *block_k_cache, T *block_v_cache,
                                               const T *scratch_k_cache, const T *scratch_v_cache,
                                               const int64_t *block_tables, const int64_t *real_slot_ids,
                                               const int64_t *scratch_slot_ids, size_t seq_len,
                                               size_t max_blocks_per_slot, size_t scratch_maxseq,
                                               size_t block_size, size_t n_kv_heads, size_t head_dim,
                                               size_t batch_size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t elems_per_token = n_kv_heads * head_dim;
    size_t total = batch_size * seq_len * elems_per_token;
    if (idx >= total) return;

    size_t elem = idx % elems_per_token;
    size_t token = (idx / elems_per_token) % seq_len;
    size_t b = idx / (seq_len * elems_per_token);
    size_t kv_h = elem / head_dim;
    size_t d = elem - kv_h * head_dim;

    int64_t real_slot = real_slot_ids[b];
    int64_t scratch_slot = scratch_slot_ids[b];
    if (real_slot < 0 || scratch_slot < 0) return;
    size_t logical_block = token / block_size;
    size_t block_offset = token % block_size;
    int64_t physical_block = block_tables[(size_t)real_slot * max_blocks_per_slot + logical_block];
    if (physical_block < 0) return;

    size_t src = (((size_t)scratch_slot * scratch_maxseq + token) * n_kv_heads + kv_h) * head_dim + d;
    size_t dst = (((size_t)physical_block * block_size + block_offset) * n_kv_heads + kv_h) * head_dim + d;
    block_k_cache[dst] = scratch_k_cache[src];
    block_v_cache[dst] = scratch_v_cache[src];
}

template <typename T>
__global__ void rope_and_scatter_kv_paged_decode_kernel(T *q, T *k, const T *v,
                                                        T *block_k_cache, T *block_v_cache,
                                                        const int64_t *block_tables,
                                                        const int64_t *pos_ids,
                                                        const int64_t *slot_ids,
                                                        const int64_t *positions,
                                                        float theta, size_t batch_size,
                                                        size_t max_blocks_per_slot, size_t block_size,
                                                        size_t n_heads, size_t n_kv_heads,
                                                        size_t head_dim, size_t max_heads) {
    size_t half_dim = head_dim / 2;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * max_heads * half_dim;
    if (idx >= total) return;

    size_t j = idx % half_dim;
    size_t h = (idx / half_dim) % max_heads;
    size_t b = idx / (half_dim * max_heads);

    int64_t pos_i64 = pos_ids[b];
    int64_t slot_i64 = slot_ids[b];
    int64_t cache_pos_i64 = positions[b];
    if (pos_i64 < 0 || slot_i64 < 0 || cache_pos_i64 < 0) return;

    float freq = (float)pos_i64 / powf(theta, (2.f * (float)j) / (float)head_dim);
    float cos_f, sin_f;
    __sincosf(freq, &sin_f, &cos_f);

    if (h < n_heads) {
        size_t q_base = (b * n_heads + h) * head_dim;
        float a = kv_to_f(q[q_base + j]);
        float c = kv_to_f(q[q_base + j + half_dim]);
        q[q_base + j] = kv_from_f<T>(a * cos_f - c * sin_f);
        q[q_base + j + half_dim] = kv_from_f<T>(c * cos_f + a * sin_f);
    }

    if (h < n_kv_heads) {
        size_t kv_base = (b * n_kv_heads + h) * head_dim;
        float a = kv_to_f(k[kv_base + j]);
        float c = kv_to_f(k[kv_base + j + half_dim]);
        T rk0 = kv_from_f<T>(a * cos_f - c * sin_f);
        T rk1 = kv_from_f<T>(c * cos_f + a * sin_f);
        k[kv_base + j] = rk0;
        k[kv_base + j + half_dim] = rk1;

        size_t slot = (size_t)slot_i64;
        size_t cache_pos = (size_t)cache_pos_i64;
        size_t logical_block = cache_pos / block_size;
        size_t block_offset = cache_pos % block_size;
        int64_t physical_block = block_tables[slot * max_blocks_per_slot + logical_block];
        if (physical_block < 0) return;
        size_t cache_base = (((size_t)physical_block * block_size + block_offset) * n_kv_heads + h) * head_dim;
        block_k_cache[cache_base + j] = rk0;
        block_k_cache[cache_base + j + half_dim] = rk1;
        block_v_cache[cache_base + j] = v[kv_base + j];
        block_v_cache[cache_base + j + half_dim] = v[kv_base + j + half_dim];
    }
}

namespace llaisys::ops::nvidia {
void scatter_kv_decode(std::byte *k_cache, std::byte *v_cache, const std::byte *k, const std::byte *v,
                       const int64_t *slot_ids, const int64_t *positions, llaisysDataType_t type,
                       size_t batch_size, size_t maxseq, size_t n_kv_heads, size_t head_dim) {
    size_t elems_per_batch = n_kv_heads * head_dim;
    size_t total = batch_size * elems_per_batch;
    int threads = 256;
    dim3 blocks((unsigned)((total + threads - 1) / threads), 1, 1);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        scatter_kv_decode_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(k_cache), reinterpret_cast<float *>(v_cache), reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v), slot_ids, positions, maxseq, n_kv_heads, head_dim, batch_size,
            elems_per_batch);
        break;
    case LLAISYS_DTYPE_BF16:
        scatter_kv_decode_kernel<__nv_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(k_cache), reinterpret_cast<__nv_bfloat16 *>(v_cache),
            reinterpret_cast<const __nv_bfloat16 *>(k), reinterpret_cast<const __nv_bfloat16 *>(v), slot_ids,
            positions, maxseq, n_kv_heads, head_dim, batch_size, elems_per_batch);
        break;
    case LLAISYS_DTYPE_F16:
        scatter_kv_decode_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<__half *>(k_cache), reinterpret_cast<__half *>(v_cache),
            reinterpret_cast<const __half *>(k), reinterpret_cast<const __half *>(v), slot_ids, positions, maxseq,
            n_kv_heads, head_dim, batch_size, elems_per_batch);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for scatter_kv_decode");
    }
}

void rope_and_scatter_kv_decode(std::byte *q, std::byte *k, const std::byte *v,
                                std::byte *k_cache, std::byte *v_cache,
                                const int64_t *pos_ids, const int64_t *slot_ids,
                                const int64_t *positions, float theta, llaisysDataType_t type,
                                size_t batch_size, size_t maxseq, size_t n_heads,
                                size_t n_kv_heads, size_t head_dim) {
    size_t max_heads = n_heads > n_kv_heads ? n_heads : n_kv_heads;
    size_t total = batch_size * max_heads * (head_dim / 2);
    int threads = 256;
    dim3 blocks((unsigned)((total + threads - 1) / threads), 1, 1);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_and_scatter_kv_decode_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(q), reinterpret_cast<float *>(k), reinterpret_cast<const float *>(v),
            reinterpret_cast<float *>(k_cache), reinterpret_cast<float *>(v_cache), pos_ids, slot_ids, positions,
            theta, batch_size, maxseq, n_heads, n_kv_heads, head_dim, max_heads);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_and_scatter_kv_decode_kernel<__nv_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(q), reinterpret_cast<__nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v), reinterpret_cast<__nv_bfloat16 *>(k_cache),
            reinterpret_cast<__nv_bfloat16 *>(v_cache), pos_ids, slot_ids, positions, theta, batch_size, maxseq,
            n_heads, n_kv_heads, head_dim, max_heads);
        break;
    case LLAISYS_DTYPE_F16:
        rope_and_scatter_kv_decode_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<__half *>(q), reinterpret_cast<__half *>(k), reinterpret_cast<const __half *>(v),
            reinterpret_cast<__half *>(k_cache), reinterpret_cast<__half *>(v_cache), pos_ids, slot_ids, positions,
            theta, batch_size, maxseq, n_heads, n_kv_heads, head_dim, max_heads);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for rope_and_scatter_kv_decode");
    }
}

void copy_kv_slots_to_blocks(std::byte *block_k_cache, std::byte *block_v_cache,
                             const std::byte *scratch_k_cache, const std::byte *scratch_v_cache,
                             const int64_t *block_tables, const int64_t *real_slot_ids,
                             const int64_t *scratch_slot_ids, llaisysDataType_t type,
                             size_t batch_size, size_t seq_len, size_t max_blocks_per_slot,
                             size_t scratch_maxseq, size_t block_size,
                             size_t n_kv_heads, size_t head_dim) {
    size_t total = batch_size * seq_len * n_kv_heads * head_dim;
    int threads = 256;
    dim3 blocks((unsigned)((total + threads - 1) / threads), 1, 1);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        copy_kv_slots_to_blocks_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(block_k_cache), reinterpret_cast<float *>(block_v_cache),
            reinterpret_cast<const float *>(scratch_k_cache), reinterpret_cast<const float *>(scratch_v_cache),
            block_tables, real_slot_ids, scratch_slot_ids, seq_len, max_blocks_per_slot, scratch_maxseq,
            block_size, n_kv_heads, head_dim, batch_size);
        break;
    case LLAISYS_DTYPE_BF16:
        copy_kv_slots_to_blocks_kernel<__nv_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(block_k_cache), reinterpret_cast<__nv_bfloat16 *>(block_v_cache),
            reinterpret_cast<const __nv_bfloat16 *>(scratch_k_cache), reinterpret_cast<const __nv_bfloat16 *>(scratch_v_cache),
            block_tables, real_slot_ids, scratch_slot_ids, seq_len, max_blocks_per_slot, scratch_maxseq,
            block_size, n_kv_heads, head_dim, batch_size);
        break;
    case LLAISYS_DTYPE_F16:
        copy_kv_slots_to_blocks_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<__half *>(block_k_cache), reinterpret_cast<__half *>(block_v_cache),
            reinterpret_cast<const __half *>(scratch_k_cache), reinterpret_cast<const __half *>(scratch_v_cache),
            block_tables, real_slot_ids, scratch_slot_ids, seq_len, max_blocks_per_slot, scratch_maxseq,
            block_size, n_kv_heads, head_dim, batch_size);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for copy_kv_slots_to_blocks");
    }
}

void rope_and_scatter_kv_paged_decode(std::byte *q, std::byte *k, const std::byte *v,
                                      std::byte *block_k_cache, std::byte *block_v_cache,
                                      const int64_t *block_tables, const int64_t *pos_ids,
                                      const int64_t *slot_ids, const int64_t *positions,
                                      float theta, llaisysDataType_t type, size_t batch_size,
                                      size_t max_blocks_per_slot, size_t block_size,
                                      size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    size_t max_heads = n_heads > n_kv_heads ? n_heads : n_kv_heads;
    size_t total = batch_size * max_heads * (head_dim / 2);
    int threads = 256;
    dim3 blocks((unsigned)((total + threads - 1) / threads), 1, 1);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_and_scatter_kv_paged_decode_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(q), reinterpret_cast<float *>(k), reinterpret_cast<const float *>(v),
            reinterpret_cast<float *>(block_k_cache), reinterpret_cast<float *>(block_v_cache), block_tables,
            pos_ids, slot_ids, positions, theta, batch_size, max_blocks_per_slot, block_size,
            n_heads, n_kv_heads, head_dim, max_heads);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_and_scatter_kv_paged_decode_kernel<__nv_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(q), reinterpret_cast<__nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v), reinterpret_cast<__nv_bfloat16 *>(block_k_cache),
            reinterpret_cast<__nv_bfloat16 *>(block_v_cache), block_tables, pos_ids, slot_ids, positions,
            theta, batch_size, max_blocks_per_slot, block_size, n_heads, n_kv_heads, head_dim, max_heads);
        break;
    case LLAISYS_DTYPE_F16:
        rope_and_scatter_kv_paged_decode_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<__half *>(q), reinterpret_cast<__half *>(k), reinterpret_cast<const __half *>(v),
            reinterpret_cast<__half *>(block_k_cache), reinterpret_cast<__half *>(block_v_cache), block_tables,
            pos_ids, slot_ids, positions, theta, batch_size, max_blocks_per_slot, block_size,
            n_heads, n_kv_heads, head_dim, max_heads);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for rope_and_scatter_kv_paged_decode");
    }
}

void split_qkv_decode(std::byte *q, std::byte *k, std::byte *v, const std::byte *qkv,
                      llaisysDataType_t type, size_t batch_size, size_t q_dim,
                      size_t k_dim, size_t v_dim) {
    size_t total_dim = q_dim + k_dim + v_dim;
    size_t total = batch_size * total_dim;
    int threads = 256;
    dim3 blocks((unsigned)((total + threads - 1) / threads), 1, 1);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        split_qkv_decode_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(q), reinterpret_cast<float *>(k), reinterpret_cast<float *>(v),
            reinterpret_cast<const float *>(qkv), batch_size, q_dim, k_dim, v_dim, total_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        split_qkv_decode_kernel<__nv_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(q), reinterpret_cast<__nv_bfloat16 *>(k),
            reinterpret_cast<__nv_bfloat16 *>(v), reinterpret_cast<const __nv_bfloat16 *>(qkv), batch_size,
            q_dim, k_dim, v_dim, total_dim);
        break;
    case LLAISYS_DTYPE_F16:
        split_qkv_decode_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<__half *>(q), reinterpret_cast<__half *>(k), reinterpret_cast<__half *>(v),
            reinterpret_cast<const __half *>(qkv), batch_size, q_dim, k_dim, v_dim, total_dim);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for split_qkv_decode");
    }
}

void split_gate_up_decode(std::byte *gate, std::byte *up, const std::byte *gate_up,
                          llaisysDataType_t type, size_t batch_size, size_t dim) {
    size_t total = batch_size * dim * 2;
    int threads = 256;
    dim3 blocks((unsigned)((total + threads - 1) / threads), 1, 1);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        split_gate_up_decode_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(gate), reinterpret_cast<float *>(up),
            reinterpret_cast<const float *>(gate_up), batch_size, dim);
        break;
    case LLAISYS_DTYPE_BF16:
        split_gate_up_decode_kernel<__nv_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(gate), reinterpret_cast<__nv_bfloat16 *>(up),
            reinterpret_cast<const __nv_bfloat16 *>(gate_up), batch_size, dim);
        break;
    case LLAISYS_DTYPE_F16:
        split_gate_up_decode_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<__half *>(gate), reinterpret_cast<__half *>(up),
            reinterpret_cast<const __half *>(gate_up), batch_size, dim);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for split_gate_up_decode");
    }
}
} // namespace llaisys::ops::nvidia
