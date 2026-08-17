#pragma once
#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::ascend {
void argmax(void *max_idx, void *max_val, const void *vals, llaisysDataType_t dtype, size_t numel);
}
