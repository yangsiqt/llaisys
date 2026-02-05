#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// RoPE: Rotary Position Embedding
// Input shape: [seq_len, n_heads, head_dim]
// pos_ids shape: [seq_len] (int64)
// Split head_dim into two halves [a, b]
// For each position p and dimension j:
//   phi_{p,j} = p / theta^(2j/d)
//   a'_{p,j} = a_{p,j} * cos(phi) - b_{p,j} * sin(phi)
//   b'_{p,j} = b_{p,j} * cos(phi) + a_{p,j} * sin(phi)

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta,
           size_t seq_len, size_t n_heads, size_t head_dim) {
    
    size_t half_dim = head_dim / 2;
    
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        // For half precision, use float for computation
        for (size_t s = 0; s < seq_len; s++) {
            float pos = static_cast<float>(pos_ids[s]);
            
            for (size_t h = 0; h < n_heads; h++) {
                for (size_t j = 0; j < half_dim; j++) {
                    // Compute frequency: phi = pos / theta^(2j/d)
                    float freq_exp = (2.0f * j) / head_dim;
                    float freq = pos / std::pow(theta, freq_exp);
                    float cos_freq = std::cos(freq);
                    float sin_freq = std::sin(freq);
                    
                    // Get input values
                    size_t idx = s * n_heads * head_dim + h * head_dim;
                    float a = llaisys::utils::cast<float>(in[idx + j]);
                    float b = llaisys::utils::cast<float>(in[idx + j + half_dim]);
                    
                    // Apply rotation
                    float a_prime = a * cos_freq - b * sin_freq;
                    float b_prime = b * cos_freq + a * sin_freq;
                    
                    out[idx + j] = llaisys::utils::cast<T>(a_prime);
                    out[idx + j + half_dim] = llaisys::utils::cast<T>(b_prime);
                }
            }
        }
    } else {
        // For full precision
        for (size_t s = 0; s < seq_len; s++) {
            float pos = static_cast<float>(pos_ids[s]);
            
            for (size_t h = 0; h < n_heads; h++) {
                for (size_t j = 0; j < half_dim; j++) {
                    // Compute frequency: phi = pos / theta^(2j/d)
                    float freq_exp = (2.0f * j) / head_dim;
                    float freq = pos / std::pow(theta, freq_exp);
                    float cos_freq = std::cos(freq);
                    float sin_freq = std::sin(freq);
                    
                    // Get input values
                    size_t idx = s * n_heads * head_dim + h * head_dim;
                    float a = static_cast<float>(in[idx + j]);
                    float b = static_cast<float>(in[idx + j + half_dim]);
                    
                    // Apply rotation
                    float a_prime = a * cos_freq - b * sin_freq;
                    float b_prime = b * cos_freq + a * sin_freq;
                    
                    out[idx + j] = static_cast<T>(a_prime);
                    out[idx + j + half_dim] = static_cast<T>(b_prime);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
          llaisysDataType_t type, size_t seq_len, size_t n_heads, size_t head_dim) {
    
    const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                    reinterpret_cast<const float *>(in),
                    pos_ptr, theta, seq_len, n_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                    reinterpret_cast<const llaisys::bf16_t *>(in),
                    pos_ptr, theta, seq_len, n_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                    reinterpret_cast<const llaisys::fp16_t *>(in),
                    pos_ptr, theta, seq_len, n_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu

