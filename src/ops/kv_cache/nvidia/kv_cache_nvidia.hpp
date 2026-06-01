#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void scatter_kv_decode(std::byte *k_cache, std::byte *v_cache, const std::byte *k, const std::byte *v,
                       const int64_t *slot_ids, const int64_t *positions, llaisysDataType_t type,
                       size_t batch_size, size_t maxseq, size_t n_kv_heads, size_t head_dim);
void rope_and_scatter_kv_decode(std::byte *q, std::byte *k, const std::byte *v,
                                std::byte *k_cache, std::byte *v_cache,
                                const int64_t *pos_ids, const int64_t *slot_ids,
                                const int64_t *positions, float theta, llaisysDataType_t type,
                                size_t batch_size, size_t maxseq, size_t n_heads,
                                size_t n_kv_heads, size_t head_dim);
void copy_kv_slots_to_blocks(std::byte *block_k_cache, std::byte *block_v_cache,
                             const std::byte *scratch_k_cache, const std::byte *scratch_v_cache,
                             const int64_t *block_tables, const int64_t *real_slot_ids,
                             const int64_t *scratch_slot_ids, llaisysDataType_t type,
                             size_t batch_size, size_t seq_len, size_t max_blocks_per_slot,
                             size_t scratch_maxseq, size_t block_size,
                             size_t n_kv_heads, size_t head_dim);
void rope_and_scatter_kv_paged_decode(std::byte *q, std::byte *k, const std::byte *v,
                                      std::byte *block_k_cache, std::byte *block_v_cache,
                                      const int64_t *block_tables, const int64_t *pos_ids,
                                      const int64_t *slot_ids, const int64_t *positions,
                                      float theta, llaisysDataType_t type, size_t batch_size,
                                      size_t max_blocks_per_slot, size_t block_size,
                                      size_t n_heads, size_t n_kv_heads, size_t head_dim);
void split_qkv_decode(std::byte *q, std::byte *k, std::byte *v, const std::byte *qkv,
                      llaisysDataType_t type, size_t batch_size, size_t q_dim,
                      size_t k_dim, size_t v_dim);
void split_gate_up_decode(std::byte *gate, std::byte *up, const std::byte *gate_up,
                          llaisysDataType_t type, size_t batch_size, size_t dim);
}
