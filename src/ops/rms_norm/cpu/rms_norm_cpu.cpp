#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// RMS Norm: y_i = (w_i * x_i) / sqrt(mean(x^2) + eps)
// For each row: normalize by RMS then scale by weight

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps,
               size_t batch_size, size_t hidden_size) {
    
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        // For half precision, use float for computation
        for (size_t b = 0; b < batch_size; b++) {
            // Compute mean of squares
            float mean_sq = 0.0f;
            for (size_t i = 0; i < hidden_size; i++) {
                float val = llaisys::utils::cast<float>(in[b * hidden_size + i]);
                mean_sq += val * val;
            }
            mean_sq /= hidden_size;
            
            // Compute rsqrt(mean_sq + eps)
            float rms_inv = 1.0f / std::sqrt(mean_sq + eps);
            
            // Normalize and scale by weight
            for (size_t i = 0; i < hidden_size; i++) {
                float val = llaisys::utils::cast<float>(in[b * hidden_size + i]);
                float w = llaisys::utils::cast<float>(weight[i]);
                float result = w * val * rms_inv;
                out[b * hidden_size + i] = llaisys::utils::cast<T>(result);
            }
        }
    } else {
        // For full precision
        for (size_t b = 0; b < batch_size; b++) {
            // Compute mean of squares
            float mean_sq = 0.0f;
            for (size_t i = 0; i < hidden_size; i++) {
                float val = static_cast<float>(in[b * hidden_size + i]);
                mean_sq += val * val;
            }
            mean_sq /= hidden_size;
            
            // Compute rsqrt(mean_sq + eps)
            float rms_inv = 1.0f / std::sqrt(mean_sq + eps);
            
            // Normalize and scale by weight
            for (size_t i = 0; i < hidden_size; i++) {
                T val = in[b * hidden_size + i];
                T w = weight[i];
                out[b * hidden_size + i] = static_cast<T>(static_cast<float>(w) * static_cast<float>(val) * rms_inv);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              llaisysDataType_t type, size_t batch_size, size_t hidden_size) {
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                        reinterpret_cast<const float *>(in),
                        reinterpret_cast<const float *>(weight),
                        eps, batch_size, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                        reinterpret_cast<const llaisys::bf16_t *>(in),
                        reinterpret_cast<const llaisys::bf16_t *>(weight),
                        eps, batch_size, hidden_size);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                        reinterpret_cast<const llaisys::fp16_t *>(in),
                        reinterpret_cast<const llaisys::fp16_t *>(weight),
                        eps, batch_size, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu

