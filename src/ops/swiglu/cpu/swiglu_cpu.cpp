#include <immintrin.h>
#include <omp.h>
#include <cmath>

#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"

// SwiGLU: out_i = up_i * SiLU(gate_i)
// SiLU(x) = x / (1 + exp(-x))

static inline __m512 avx512_silu(__m512 x) {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // Fast approximation: compute exp(-x) via polynomial or use precise scalar fallback
    float tmp_in[16], tmp_out[16];
    _mm512_storeu_ps(tmp_in, x);
    for (int i = 0; i < 16; i++)
        tmp_out[i] = tmp_in[i] / (1.0f + std::exp(-tmp_in[i]));
    return _mm512_loadu_ps(tmp_out);
}

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numel; i++) {
            float gate_val = llaisys::utils::cast<float>(gate[i]);
            float up_val = llaisys::utils::cast<float>(up[i]);
            float silu = gate_val / (1.0f + std::exp(-gate_val));
            out[i] = llaisys::utils::cast<T>(up_val * silu);
        }
    } else {
        const float *g = reinterpret_cast<const float *>(gate);
        const float *u = reinterpret_cast<const float *>(up);
        float *o = reinterpret_cast<float *>(out);

        #pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < numel; idx += 16) {
            size_t end = std::min(idx + (size_t)16, numel);
            if (end - idx == 16) {
                __m512 vg = _mm512_loadu_ps(g + idx);
                __m512 vu = _mm512_loadu_ps(u + idx);
                __m512 vsilu = avx512_silu(vg);
                __m512 vresult = _mm512_mul_ps(vu, vsilu);
                _mm512_storeu_ps(o + idx, vresult);
            } else {
                for (size_t j = idx; j < end; j++) {
                    float gv = g[j];
                    float uv = u[j];
                    o[j] = uv * (gv / (1.0f + std::exp(-gv)));
                }
            }
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
