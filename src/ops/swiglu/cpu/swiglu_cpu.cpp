#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// SwiGLU: out_i = up_i * (gate_i / (1 + exp(-gate_i)))
// This is element-wise SiLU(gate) * up

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        // For half precision, use float for computation
        for (size_t i = 0; i < numel; i++) {
            float gate_val = llaisys::utils::cast<float>(gate[i]);
            float up_val = llaisys::utils::cast<float>(up[i]);
            
            // SiLU(x) = x / (1 + exp(-x)) = x * sigmoid(x)
            float silu = gate_val / (1.0f + std::exp(-gate_val));
            float result = up_val * silu;
            
            out[i] = llaisys::utils::cast<T>(result);
        }
    } else {
        // For full precision
        for (size_t i = 0; i < numel; i++) {
            float gate_val = static_cast<float>(gate[i]);
            float up_val = static_cast<float>(up[i]);
            
            // SiLU(x) = x / (1 + exp(-x))
            float silu = gate_val / (1.0f + std::exp(-gate_val));
            float result = up_val * silu;
            
            out[i] = static_cast<T>(result);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, 
            llaisysDataType_t type, size_t numel) {
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out),
                      reinterpret_cast<const float *>(gate),
                      reinterpret_cast<const float *>(up),
                      numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out),
                      reinterpret_cast<const llaisys::bf16_t *>(gate),
                      reinterpret_cast<const llaisys::bf16_t *>(up),
                      numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out),
                      reinterpret_cast<const llaisys::fp16_t *>(gate),
                      reinterpret_cast<const llaisys::fp16_t *>(up),
                      numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu

