#pragma once
#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::ascend {
void swiglu(void *out, const void *gate, const void *up, llaisysDataType_t dtype, size_t numel);
}
