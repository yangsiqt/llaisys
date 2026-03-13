#include <immintrin.h>
#include <omp.h>
#include <cmath>

#include "add_cpu.hpp"
#include "../../../utils.hpp"

namespace llaisys {
namespace ops {
namespace cpu {

template<typename T>
void add_cpu_impl(T* out, const T* a, const T* b, size_t numel) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numel; i++) {
            float a_val = llaisys::utils::cast<float>(a[i]);
            float b_val = llaisys::utils::cast<float>(b[i]);
            out[i] = llaisys::utils::cast<T>(a_val + b_val);
        }
    } else {
        const float *fa = reinterpret_cast<const float *>(a);
        const float *fb = reinterpret_cast<const float *>(b);
        float *fo = reinterpret_cast<float *>(out);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numel; i += 16) {
            size_t end = std::min(i + (size_t)16, numel);
            if (end - i == 16) {
                __m512 va = _mm512_loadu_ps(fa + i);
                __m512 vb = _mm512_loadu_ps(fb + i);
                _mm512_storeu_ps(fo + i, _mm512_add_ps(va, vb));
            } else {
                for (size_t j = i; j < end; j++)
                    fo[j] = fa[j] + fb[j];
            }
        }
    }
}

void add_cpu(const tensor_t& out, const tensor_t& a, const tensor_t& b) {
    size_t numel = out->numel();
    
    switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            add_cpu_impl<float>(
                reinterpret_cast<float*>(out->data()),
                reinterpret_cast<const float*>(a->data()),
                reinterpret_cast<const float*>(b->data()),
                numel
            );
            break;
        case LLAISYS_DTYPE_F16:
            add_cpu_impl<llaisys::fp16_t>(
                reinterpret_cast<llaisys::fp16_t*>(out->data()),
                reinterpret_cast<const llaisys::fp16_t*>(a->data()),
                reinterpret_cast<const llaisys::fp16_t*>(b->data()),
                numel
            );
            break;
        case LLAISYS_DTYPE_BF16:
            add_cpu_impl<llaisys::bf16_t>(
                reinterpret_cast<llaisys::bf16_t*>(out->data()),
                reinterpret_cast<const llaisys::bf16_t*>(a->data()),
                reinterpret_cast<const llaisys::bf16_t*>(b->data()),
                numel
            );
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace cpu
} // namespace ops
} // namespace llaisys
