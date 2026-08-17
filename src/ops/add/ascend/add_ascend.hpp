#pragma once
#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::ascend {
void add(void *out, const void *a, const void *b, llaisysDataType_t dtype, size_t numel);
}
