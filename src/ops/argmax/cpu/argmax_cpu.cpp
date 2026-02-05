#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    T max_value;
    int64_t max_index = 0;
    
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        // For half precision types, convert to float for comparison
        float max_float = llaisys::utils::cast<float>(vals[0]);
        for (size_t i = 1; i < numel; i++) {
            float val_float = llaisys::utils::cast<float>(vals[i]);
            if (val_float > max_float) {
                max_float = val_float;
                max_index = i;
            }
        }
        max_value = llaisys::utils::cast<T>(max_float);
    } else {
        // For full precision types
        max_value = vals[0];
        for (size_t i = 1; i < numel; i++) {
            if (vals[i] > max_value) {
                max_value = vals[i];
                max_index = i;
            }
        }
    }
    
    max_idx[0] = max_index;
    max_val[0] = max_value;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    int64_t *idx_ptr = reinterpret_cast<int64_t *>(max_idx);
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(idx_ptr, reinterpret_cast<float *>(max_val), 
                      reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(idx_ptr, reinterpret_cast<llaisys::bf16_t *>(max_val),
                      reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(idx_ptr, reinterpret_cast<llaisys::fp16_t *>(max_val),
                      reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu

