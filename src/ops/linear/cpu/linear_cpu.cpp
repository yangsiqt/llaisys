#ifdef LLAISYS_USE_OPENBLAS
#include <immintrin.h>
#include <omp.h>
#include <cblas.h>
#endif

#include <cstring>
#include <vector>

#include "linear_cpu.hpp"
#include "../../../utils.hpp"

// Linear: out = in @ weight^T + bias
// out: [batch_size, out_features]
// in: [batch_size, in_features]
// weight: [out_features, in_features] (NOT transposed)
// bias: [out_features] (optional)

#ifdef LLAISYS_USE_OPENBLAS
static void linear_f32(float *out, const float *in, const float *weight, const float *bias,
                       size_t batch_size, size_t in_features, size_t out_features) {
    int M = static_cast<int>(batch_size);
    int N = static_cast<int>(out_features);
    int K = static_cast<int>(in_features);

    if (bias != nullptr) {
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < M; b++)
            memcpy(out + b * N, bias, N * sizeof(float));
    }

    if (M == 1) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    N, K, 1.0f, weight, K, in, 1,
                    bias ? 1.0f : 0.0f, out, 1);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K, 1.0f, in, K, weight, K,
                    bias ? 1.0f : 0.0f, out, N);
    }
}
#else
// Scalar fallback when OpenBLAS is not available (e.g. Windows CI)
static void linear_f32(float *out, const float *in, const float *weight, const float *bias,
                       size_t batch_size, size_t in_features, size_t out_features) {
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < out_features; o++) {
            float sum = 0.0f;
            for (size_t i = 0; i < in_features; i++)
                sum += in[b * in_features + i] * weight[o * in_features + i];
            if (bias != nullptr)
                sum += bias[o];
            out[b * out_features + o] = sum;
        }
    }
}
#endif

#ifdef LLAISYS_USE_OPENBLAS
static inline __m512 bf16x16_to_f32x16(const llaisys::bf16_t *src) {
    __m256i bf16_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
    __m512i i32_vals = _mm512_cvtepu16_epi32(bf16_vals);
    __m512i shifted = _mm512_slli_epi32(i32_vals, 16);
    return _mm512_castsi512_ps(shifted);
}

static void bf16_to_f32_avx512(float *dst, const llaisys::bf16_t *src, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i, bf16x16_to_f32x16(src + i));
    }
    for (; i < n; i++) {
        dst[i] = llaisys::utils::cast<float>(src[i]);
    }
}

static void f32_to_bf16_avx512(llaisys::bf16_t *dst, const float *src, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512i f32_bits = _mm512_castps_si512(_mm512_loadu_ps(src + i));
        __m512i rounding_bias = _mm512_add_epi32(
            _mm512_set1_epi32(0x00007FFF),
            _mm512_and_si512(_mm512_srli_epi32(f32_bits, 16), _mm512_set1_epi32(1))
        );
        __m512i rounded = _mm512_add_epi32(f32_bits, rounding_bias);
        __m512i shifted = _mm512_srli_epi32(rounded, 16);
        __m256i packed = _mm512_cvtepi32_epi16(shifted);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), packed);
    }
    for (; i < n; i++) {
        dst[i] = llaisys::utils::cast<llaisys::bf16_t>(src[i]);
    }
}

// Fused bf16 matrix-vector multiply: out[o] = sum_i(weight[o,i] * in_f32[i])
static void bf16_gemv_fused(float *out_f32, const llaisys::bf16_t *weight,
                            const float *in_f32, const llaisys::bf16_t *bias,
                            size_t in_features, size_t out_features) {
    #pragma omp parallel for schedule(static)
    for (size_t o = 0; o < out_features; o++) {
        const llaisys::bf16_t *w_row = weight + o * in_features;
        __m512 acc = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= in_features; i += 16) {
            __m512 w_f32 = bf16x16_to_f32x16(w_row + i);
            __m512 in_vec = _mm512_loadu_ps(in_f32 + i);
            acc = _mm512_fmadd_ps(w_f32, in_vec, acc);
        }
        float sum = _mm512_reduce_add_ps(acc);
        for (; i < in_features; i++) {
            sum += llaisys::utils::cast<float>(w_row[i]) * in_f32[i];
        }
        if (bias != nullptr)
            sum += llaisys::utils::cast<float>(bias[o]);
        out_f32[o] = sum;
    }
}

static void linear_bf16(llaisys::bf16_t *out, const llaisys::bf16_t *in,
                        const llaisys::bf16_t *weight, const llaisys::bf16_t *bias,
                        size_t batch_size, size_t in_features, size_t out_features) {

    if (batch_size == 1) {
        std::vector<float> in_f32(in_features);
        bf16_to_f32_avx512(in_f32.data(), in, in_features);

        std::vector<float> out_f32(out_features);
        bf16_gemv_fused(out_f32.data(), weight, in_f32.data(), bias, in_features, out_features);

        f32_to_bf16_avx512(out, out_f32.data(), out_features);
    } else {
        size_t in_elems = batch_size * in_features;
        size_t w_elems = out_features * in_features;
        size_t out_elems = batch_size * out_features;

        std::vector<float> in_f32(in_elems);
        std::vector<float> w_f32(w_elems);
        std::vector<float> out_f32(out_elems, 0.0f);

        bf16_to_f32_avx512(in_f32.data(), in, in_elems);
        bf16_to_f32_avx512(w_f32.data(), weight, w_elems);

        int M = static_cast<int>(batch_size);
        int N = static_cast<int>(out_features);
        int K = static_cast<int>(in_features);

        if (bias != nullptr) {
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < M; b++) {
                for (int j = 0; j < N; j++)
                    out_f32[b * N + j] = llaisys::utils::cast<float>(bias[j]);
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0f, in_f32.data(), K, w_f32.data(), K,
                        1.0f, out_f32.data(), N);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0f, in_f32.data(), K, w_f32.data(), K,
                        0.0f, out_f32.data(), N);
        }

        f32_to_bf16_avx512(out, out_f32.data(), out_elems);
    }
}

static void linear_fp16(llaisys::fp16_t *out, const llaisys::fp16_t *in,
                        const llaisys::fp16_t *weight, const llaisys::fp16_t *bias,
                        size_t batch_size, size_t in_features, size_t out_features) {
    size_t in_elems = batch_size * in_features;
    size_t w_elems = out_features * in_features;
    size_t out_elems = batch_size * out_features;

    std::vector<float> in_f32(in_elems);
    std::vector<float> w_f32(w_elems);
    std::vector<float> out_f32(out_elems, 0.0f);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < in_elems; i++)
        in_f32[i] = llaisys::utils::cast<float>(in[i]);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < w_elems; i++)
        w_f32[i] = llaisys::utils::cast<float>(weight[i]);

    int M = static_cast<int>(batch_size);
    int N = static_cast<int>(out_features);
    int K = static_cast<int>(in_features);

    if (M == 1) {
        if (bias != nullptr) {
            for (int j = 0; j < N; j++)
                out_f32[j] = llaisys::utils::cast<float>(bias[j]);
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        N, K, 1.0f, w_f32.data(), K, in_f32.data(), 1,
                        1.0f, out_f32.data(), 1);
        } else {
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        N, K, 1.0f, w_f32.data(), K, in_f32.data(), 1,
                        0.0f, out_f32.data(), 1);
        }
    } else {
        if (bias != nullptr) {
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < M; b++) {
                for (int j = 0; j < N; j++)
                    out_f32[b * N + j] = llaisys::utils::cast<float>(bias[j]);
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0f, in_f32.data(), K, w_f32.data(), K,
                        1.0f, out_f32.data(), N);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0f, in_f32.data(), K, w_f32.data(), K,
                        0.0f, out_f32.data(), N);
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < out_elems; i++)
        out[i] = llaisys::utils::cast<llaisys::fp16_t>(out_f32[i]);
}
#else
// Scalar fallback for bf16/fp16 when OpenBLAS is not available
template <typename T>
static void linear_half_fallback(T *out, const T *in, const T *weight, const T *bias,
                                 size_t batch_size, size_t in_features, size_t out_features) {
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < out_features; o++) {
            float sum = 0.0f;
            for (size_t i = 0; i < in_features; i++) {
                float in_val = llaisys::utils::cast<float>(in[b * in_features + i]);
                float w_val = llaisys::utils::cast<float>(weight[o * in_features + i]);
                sum += in_val * w_val;
            }
            if (bias != nullptr)
                sum += llaisys::utils::cast<float>(bias[o]);
            out[b * out_features + o] = llaisys::utils::cast<T>(sum);
        }
    }
}
#endif

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features) {
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_f32(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight),
                         bias ? reinterpret_cast<const float *>(bias) : nullptr,
                         batch_size, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
#ifdef LLAISYS_USE_OPENBLAS
        return linear_bf16(reinterpret_cast<llaisys::bf16_t *>(out),
                          reinterpret_cast<const llaisys::bf16_t *>(in),
                          reinterpret_cast<const llaisys::bf16_t *>(weight),
                          bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
                          batch_size, in_features, out_features);
#else
        return linear_half_fallback<llaisys::bf16_t>(
                          reinterpret_cast<llaisys::bf16_t *>(out),
                          reinterpret_cast<const llaisys::bf16_t *>(in),
                          reinterpret_cast<const llaisys::bf16_t *>(weight),
                          bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
                          batch_size, in_features, out_features);
#endif
    case LLAISYS_DTYPE_F16:
#ifdef LLAISYS_USE_OPENBLAS
        return linear_fp16(reinterpret_cast<llaisys::fp16_t *>(out),
                          reinterpret_cast<const llaisys::fp16_t *>(in),
                          reinterpret_cast<const llaisys::fp16_t *>(weight),
                          bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
                          batch_size, in_features, out_features);
#else
        return linear_half_fallback<llaisys::fp16_t>(
                          reinterpret_cast<llaisys::fp16_t *>(out),
                          reinterpret_cast<const llaisys::fp16_t *>(in),
                          reinterpret_cast<const llaisys::fp16_t *>(weight),
                          bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
                          batch_size, in_features, out_features);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
