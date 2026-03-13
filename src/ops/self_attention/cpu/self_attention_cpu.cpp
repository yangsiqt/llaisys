#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"

// Self-Attention with causal mask and GQA (Grouped Query Attention)
// Q: [qlen, n_heads, head_dim]
// K: [kvlen, n_kv_heads, head_dim]
// V: [kvlen, n_kv_heads, head_dim]
// Output: [qlen, n_heads, head_dim]

static inline float avx512_dot(const float *a, const float *b, size_t len) {
    __m512 sum_vec = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        sum_vec = _mm512_fmadd_ps(va, vb, sum_vec);
    }
    float sum = _mm512_reduce_add_ps(sum_vec);
    for (; i < len; i++)
        sum += a[i] * b[i];
    return sum;
}

static inline __m512 bf16x16_to_f32x16(const llaisys::bf16_t *src) {
    __m256i bf16_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
    __m512i i32_vals = _mm512_cvtepu16_epi32(bf16_vals);
    return _mm512_castsi512_ps(_mm512_slli_epi32(i32_vals, 16));
}

static void bf16_to_f32_fast(float *dst, const llaisys::bf16_t *src, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(dst + i, bf16x16_to_f32x16(src + i));
    for (; i < n; i++)
        dst[i] = llaisys::utils::cast<float>(src[i]);
}

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale,
                     size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    
    size_t n_rep = n_heads / n_kv_heads;

    if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        size_t q_elems = qlen * n_heads * head_dim;
        size_t k_elems = kvlen * n_kv_heads * head_dim;
        size_t v_elems = kvlen * n_kv_heads * head_dim;

        std::vector<float> q_f32(q_elems), k_f32(k_elems), v_f32(v_elems);
        bf16_to_f32_fast(q_f32.data(), q, q_elems);
        bf16_to_f32_fast(k_f32.data(), k, k_elems);
        bf16_to_f32_fast(v_f32.data(), v, v_elems);

        #pragma omp parallel
        {
            std::vector<float> scores(qlen * kvlen);

            #pragma omp for schedule(dynamic)
            for (size_t h = 0; h < n_heads; h++) {
                size_t kv_h = h / n_rep;

                for (size_t qi = 0; qi < qlen; qi++) {
                    const float *q_ptr = q_f32.data() + qi * n_heads * head_dim + h * head_dim;
                    size_t abs_qi = kvlen - qlen + qi;

                    for (size_t ki = 0; ki < kvlen; ki++) {
                        if (ki > abs_qi) {
                            scores[qi * kvlen + ki] = -std::numeric_limits<float>::infinity();
                            continue;
                        }
                        const float *k_ptr = k_f32.data() + ki * n_kv_heads * head_dim + kv_h * head_dim;
                        scores[qi * kvlen + ki] = avx512_dot(q_ptr, k_ptr, head_dim) * scale;
                    }

                    float max_score = -std::numeric_limits<float>::infinity();
                    for (size_t ki = 0; ki < kvlen; ki++)
                        max_score = std::max(max_score, scores[qi * kvlen + ki]);

                    float sum_exp = 0.0f;
                    for (size_t ki = 0; ki < kvlen; ki++) {
                        scores[qi * kvlen + ki] = std::exp(scores[qi * kvlen + ki] - max_score);
                        sum_exp += scores[qi * kvlen + ki];
                    }
                    float inv_sum = 1.0f / sum_exp;
                    for (size_t ki = 0; ki < kvlen; ki++)
                        scores[qi * kvlen + ki] *= inv_sum;

                    for (size_t d = 0; d < head_dim; d++) {
                        float val = 0.0f;
                        for (size_t ki = 0; ki < kvlen; ki++)
                            val += scores[qi * kvlen + ki] * v_f32[ki * n_kv_heads * head_dim + kv_h * head_dim + d];
                        attn_val[qi * n_heads * head_dim + h * head_dim + d] = llaisys::utils::cast<T>(val);
                    }
                }
            }
        }
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        size_t q_elems = qlen * n_heads * head_dim;
        size_t k_elems = kvlen * n_kv_heads * head_dim;
        size_t v_elems = kvlen * n_kv_heads * head_dim;

        std::vector<float> q_f32(q_elems), k_f32(k_elems), v_f32(v_elems);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < q_elems; i++) q_f32[i] = llaisys::utils::cast<float>(q[i]);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < k_elems; i++) k_f32[i] = llaisys::utils::cast<float>(k[i]);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < v_elems; i++) v_f32[i] = llaisys::utils::cast<float>(v[i]);

        #pragma omp parallel
        {
            std::vector<float> scores(qlen * kvlen);

            #pragma omp for schedule(dynamic)
            for (size_t h = 0; h < n_heads; h++) {
                size_t kv_h = h / n_rep;

                for (size_t qi = 0; qi < qlen; qi++) {
                    const float *q_ptr = q_f32.data() + qi * n_heads * head_dim + h * head_dim;
                    size_t abs_qi = kvlen - qlen + qi;

                    for (size_t ki = 0; ki < kvlen; ki++) {
                        if (ki > abs_qi) {
                            scores[qi * kvlen + ki] = -std::numeric_limits<float>::infinity();
                            continue;
                        }
                        const float *k_ptr = k_f32.data() + ki * n_kv_heads * head_dim + kv_h * head_dim;
                        scores[qi * kvlen + ki] = avx512_dot(q_ptr, k_ptr, head_dim) * scale;
                    }

                    float max_score = -std::numeric_limits<float>::infinity();
                    for (size_t ki = 0; ki < kvlen; ki++)
                        max_score = std::max(max_score, scores[qi * kvlen + ki]);

                    float sum_exp = 0.0f;
                    for (size_t ki = 0; ki < kvlen; ki++) {
                        scores[qi * kvlen + ki] = std::exp(scores[qi * kvlen + ki] - max_score);
                        sum_exp += scores[qi * kvlen + ki];
                    }
                    float inv_sum = 1.0f / sum_exp;
                    for (size_t ki = 0; ki < kvlen; ki++)
                        scores[qi * kvlen + ki] *= inv_sum;

                    for (size_t d = 0; d < head_dim; d++) {
                        float val = 0.0f;
                        for (size_t ki = 0; ki < kvlen; ki++)
                            val += scores[qi * kvlen + ki] * v_f32[ki * n_kv_heads * head_dim + kv_h * head_dim + d];
                        attn_val[qi * n_heads * head_dim + h * head_dim + d] = llaisys::utils::cast<T>(val);
                    }
                }
            }
        }
    } else {
        #pragma omp parallel
        {
            std::vector<float> scores(qlen * kvlen);

            #pragma omp for schedule(dynamic)
            for (size_t h = 0; h < n_heads; h++) {
                size_t kv_h = h / n_rep;

                for (size_t qi = 0; qi < qlen; qi++) {
                    const float *q_ptr = reinterpret_cast<const float *>(q) + qi * n_heads * head_dim + h * head_dim;
                    size_t abs_qi = kvlen - qlen + qi;

                    for (size_t ki = 0; ki < kvlen; ki++) {
                        if (ki > abs_qi) {
                            scores[qi * kvlen + ki] = -std::numeric_limits<float>::infinity();
                            continue;
                        }
                        const float *k_ptr = reinterpret_cast<const float *>(k) + ki * n_kv_heads * head_dim + kv_h * head_dim;
                        scores[qi * kvlen + ki] = avx512_dot(q_ptr, k_ptr, head_dim) * scale;
                    }

                    float max_score = -std::numeric_limits<float>::infinity();
                    for (size_t ki = 0; ki < kvlen; ki++)
                        max_score = std::max(max_score, scores[qi * kvlen + ki]);

                    float sum_exp = 0.0f;
                    for (size_t ki = 0; ki < kvlen; ki++) {
                        scores[qi * kvlen + ki] = std::exp(scores[qi * kvlen + ki] - max_score);
                        sum_exp += scores[qi * kvlen + ki];
                    }
                    float inv_sum = 1.0f / sum_exp;
                    for (size_t ki = 0; ki < kvlen; ki++)
                        scores[qi * kvlen + ki] *= inv_sum;

                    for (size_t d = 0; d < head_dim; d++) {
                        float val = 0.0f;
                        for (size_t ki = 0; ki < kvlen; ki++)
                            val += scores[qi * kvlen + ki] * static_cast<float>(
                                reinterpret_cast<const float *>(v)[ki * n_kv_heads * head_dim + kv_h * head_dim + d]);
                        reinterpret_cast<float *>(attn_val)[qi * n_heads * head_dim + h * head_dim + d] = val;
                    }
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, 
                    float scale, llaisysDataType_t type, 
                    size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                              reinterpret_cast<const float *>(q),
                              reinterpret_cast<const float *>(k),
                              reinterpret_cast<const float *>(v),
                              scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                              reinterpret_cast<const llaisys::bf16_t *>(q),
                              reinterpret_cast<const llaisys::bf16_t *>(k),
                              reinterpret_cast<const llaisys::bf16_t *>(v),
                              scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                              reinterpret_cast<const llaisys::fp16_t *>(q),
                              reinterpret_cast<const llaisys::fp16_t *>(k),
                              reinterpret_cast<const llaisys::fp16_t *>(v),
                              scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
