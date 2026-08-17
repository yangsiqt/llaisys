#pragma once
#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::ascend {
void rms_norm(void *out, const void *in, const void *weight, float eps, llaisysDataType_t dtype, size_t batch_size, size_t hidden_size);
}
