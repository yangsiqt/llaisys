#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

// Linear: out = in @ weight^T + bias
// out: [batch_size, out_features]
// in: [batch_size, in_features]
// weight: [out_features, in_features] (NOT transposed)
// bias: [out_features] (optional)

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t batch_size, size_t in_features, size_t out_features) {
    
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        // For half precision, use float accumulation
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t o = 0; o < out_features; o++) {
                float sum = 0.0f;
                for (size_t i = 0; i < in_features; i++) {
                    // weight[o, i] @ in[b, i]
                    float in_val = llaisys::utils::cast<float>(in[b * in_features + i]);
                    float w_val = llaisys::utils::cast<float>(weight[o * in_features + i]);
                    sum += in_val * w_val;
                }
                if (bias != nullptr) {
                    sum += llaisys::utils::cast<float>(bias[o]);
                }
                out[b * out_features + o] = llaisys::utils::cast<T>(sum);
            }
        }
    } else {
        // For full precision
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t o = 0; o < out_features; o++) {
                T sum = 0;
                for (size_t i = 0; i < in_features; i++) {
                    // weight[o, i] @ in[b, i]
                    sum += in[b * in_features + i] * weight[o * in_features + i];
                }
                if (bias != nullptr) {
                    sum += bias[o];
                }
                out[b * out_features + o] = sum;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features) {
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                      reinterpret_cast<const float *>(in),
                      reinterpret_cast<const float *>(weight),
                      bias ? reinterpret_cast<const float *>(bias) : nullptr,
                      batch_size, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                      reinterpret_cast<const llaisys::bf16_t *>(in),
                      reinterpret_cast<const llaisys::bf16_t *>(weight),
                      bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
                      batch_size, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                      reinterpret_cast<const llaisys::fp16_t *>(in),
                      reinterpret_cast<const llaisys::fp16_t *>(weight),
                      bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
                      batch_size, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu

