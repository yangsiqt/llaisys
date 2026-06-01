#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void scatter_kv_decode(tensor_t k_cache, tensor_t v_cache, tensor_t k, tensor_t v,
                       tensor_t slot_ids, tensor_t positions);
void rope_and_scatter_kv_decode(tensor_t q, tensor_t k, tensor_t v,
                                tensor_t k_cache, tensor_t v_cache,
                                tensor_t pos_ids, tensor_t slot_ids,
                                tensor_t positions, float theta);
void split_qkv_decode(tensor_t q, tensor_t k, tensor_t v, tensor_t qkv);
void split_gate_up_decode(tensor_t gate, tensor_t up, tensor_t gate_up);
}
