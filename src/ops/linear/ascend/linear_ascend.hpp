#pragma once
#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::ascend {
void linear(void *out, const void *in, const void *weight, const void *bias, llaisysDataType_t dtype, size_t batch_size, size_t in_features, size_t out_features);
}
