#pragma once
#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::ascend {
void self_attention(void *out, const void *q, const void *k, const void *v, float scale,
                    llaisysDataType_t dtype, size_t qlen, size_t kvlen,
                    size_t n_heads, size_t n_kv_heads, size_t head_dim);
}
