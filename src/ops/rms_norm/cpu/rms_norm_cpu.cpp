#include <immintrin.h>
#include <omp.h>
#include <cmath>

#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"

// RMS Norm: y_i = (w_i * x_i) / sqrt(mean(x^2) + eps)

static inline __m512 bf16x16_to_f32x16(const llaisys::bf16_t *src) {
    __m256i bf16_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
    __m512i i32_vals = _mm512_cvtepu16_epi32(bf16_vals);
    return _mm512_castsi512_ps(_mm512_slli_epi32(i32_vals, 16));
}

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps,
               size_t batch_size, size_t hidden_size) {
    
    if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        #pragma omp parallel for schedule(static)
        for (size_t b = 0; b < batch_size; b++) {
            const T *in_row = in + b * hidden_size;
            T *out_row = out + b * hidden_size;

            __m512 sum_sq_vec = _mm512_setzero_ps();
            size_t i = 0;
            for (; i + 16 <= hidden_size; i += 16) {
                __m512 v = bf16x16_to_f32x16(in_row + i);
                sum_sq_vec = _mm512_fmadd_ps(v, v, sum_sq_vec);
            }
            float mean_sq = _mm512_reduce_add_ps(sum_sq_vec);
            for (; i < hidden_size; i++) {
                float val = llaisys::utils::cast<float>(in_row[i]);
                mean_sq += val * val;
            }
            mean_sq /= hidden_size;
            float rms_inv = 1.0f / std::sqrt(mean_sq + eps);

            __m512 rms_inv_vec = _mm512_set1_ps(rms_inv);
            i = 0;
            for (; i + 16 <= hidden_size; i += 16) {
                __m512 vi = bf16x16_to_f32x16(in_row + i);
                __m512 vw = bf16x16_to_f32x16(weight + i);
                __m512 result = _mm512_mul_ps(_mm512_mul_ps(vw, vi), rms_inv_vec);
                // Convert f32 back to bf16 with rounding
                __m512i f32_bits = _mm512_castps_si512(result);
                __m512i rounding = _mm512_add_epi32(
                    _mm512_set1_epi32(0x00007FFF),
                    _mm512_and_si512(_mm512_srli_epi32(f32_bits, 16), _mm512_set1_epi32(1)));
                __m512i rounded = _mm512_add_epi32(f32_bits, rounding);
                __m256i packed = _mm512_cvtepi32_epi16(_mm512_srli_epi32(rounded, 16));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_row + i), packed);
            }
            for (; i < hidden_size; i++) {
                float val = llaisys::utils::cast<float>(in_row[i]);
                float w = llaisys::utils::cast<float>(weight[i]);
                out_row[i] = llaisys::utils::cast<T>(w * val * rms_inv);
            }
        }
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        #pragma omp parallel for schedule(static)
        for (size_t b = 0; b < batch_size; b++) {
            const T *in_row = in + b * hidden_size;
            T *out_row = out + b * hidden_size;

            float mean_sq = 0.0f;
            for (size_t i = 0; i < hidden_size; i++) {
                float val = llaisys::utils::cast<float>(in_row[i]);
                mean_sq += val * val;
            }
            mean_sq /= hidden_size;
            float rms_inv = 1.0f / std::sqrt(mean_sq + eps);

            for (size_t i = 0; i < hidden_size; i++) {
                float val = llaisys::utils::cast<float>(in_row[i]);
                float w = llaisys::utils::cast<float>(weight[i]);
                out_row[i] = llaisys::utils::cast<T>(w * val * rms_inv);
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t b = 0; b < batch_size; b++) {
            const float *in_row = reinterpret_cast<const float *>(in) + b * hidden_size;
            float *out_row = reinterpret_cast<float *>(out) + b * hidden_size;

            __m512 sum_sq_vec = _mm512_setzero_ps();
            size_t i = 0;
            for (; i + 16 <= hidden_size; i += 16) {
                __m512 v = _mm512_loadu_ps(in_row + i);
                sum_sq_vec = _mm512_fmadd_ps(v, v, sum_sq_vec);
            }
            float mean_sq = _mm512_reduce_add_ps(sum_sq_vec);
            for (; i < hidden_size; i++)
                mean_sq += in_row[i] * in_row[i];
            mean_sq /= hidden_size;
            float rms_inv = 1.0f / std::sqrt(mean_sq + eps);

            __m512 rms_inv_vec = _mm512_set1_ps(rms_inv);
            const float *w_ptr = reinterpret_cast<const float *>(weight);
            i = 0;
            for (; i + 16 <= hidden_size; i += 16) {
                __m512 vi = _mm512_loadu_ps(in_row + i);
                __m512 vw = _mm512_loadu_ps(w_ptr + i);
                _mm512_storeu_ps(out_row + i, _mm512_mul_ps(_mm512_mul_ps(vw, vi), rms_inv_vec));
            }
            for (; i < hidden_size; i++)
                out_row[i] = w_ptr[i] * in_row[i] * rms_inv;
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
