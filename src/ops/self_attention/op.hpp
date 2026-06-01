#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale,
                   size_t qlen_per_seq = 0);  // 0 means use qlen (backward compat, bs=1)

void self_attention_slots_decode(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache,
                                 tensor_t slot_ids, tensor_t seq_lens, float scale);
void self_attention_gqa_slots_decode(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache,
                                     tensor_t slot_ids, tensor_t seq_lens, float scale);
void self_attention_paged_slots_decode(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache,
                                       tensor_t block_tables, tensor_t slot_ids, tensor_t seq_lens,
                                       float scale);
void self_attention_paged_gqa_slots_decode(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache,
                                           tensor_t block_tables, tensor_t slot_ids, tensor_t seq_lens,
                                           float scale);
}
