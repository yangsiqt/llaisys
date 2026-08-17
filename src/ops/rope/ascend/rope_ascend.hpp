#pragma once
#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::ascend {
void rope(void *out, const void *in, const int64_t *pos_ids, float theta, llaisysDataType_t dtype, size_t seq_len, size_t n_heads, size_t head_dim);
}
