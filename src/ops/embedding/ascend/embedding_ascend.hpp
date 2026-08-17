#pragma once
#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::ascend {
void embedding(void *out, const void *index, const void *weight, llaisysDataType_t dtype, size_t seq_len, size_t hidden_size, size_t vocab_size);
}
