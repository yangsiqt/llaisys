#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t size);
void argmax_batch(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
                  llaisysDataType_t type, size_t batch_size, size_t width);
void argmax_batch_fast(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
                       std::byte *partial_idx, std::byte *partial_val,
                       llaisysDataType_t type, size_t batch_size, size_t width,
                       size_t num_parts);
}
