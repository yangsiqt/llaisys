#pragma once
#include "llaisys.h"
#include <cstddef>
#include <vector>
namespace llaisys::ops::ascend {
void rearrange(void *out, const void *in, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &out_strides, const std::vector<ptrdiff_t> &in_strides, llaisysDataType_t dtype);
}
