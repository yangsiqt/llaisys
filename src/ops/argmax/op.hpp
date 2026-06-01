#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
void argmax_batch(tensor_t max_idx, tensor_t max_val, tensor_t vals);
void argmax_batch_fast(tensor_t max_idx, tensor_t max_val, tensor_t vals,
                       tensor_t partial_idx, tensor_t partial_val);
}
