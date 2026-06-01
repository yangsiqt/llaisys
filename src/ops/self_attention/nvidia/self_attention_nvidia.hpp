#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t type,
                    size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads, size_t head_dim,
                    size_t qlen_per_seq);
void self_attention_slots_decode(std::byte *attn_val, const std::byte *q, const std::byte *k_cache,
                                 const std::byte *v_cache, const int64_t *slot_ids, const int64_t *seq_lens,
                                 float scale, llaisysDataType_t type, size_t batch_size, size_t maxseq,
                                 size_t n_heads, size_t n_kv_heads, size_t head_dim);
}
