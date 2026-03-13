#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t seq_len, size_t hidden_size) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < seq_len; i++) {
        int64_t idx = index[i];
        const T *src = weight + idx * hidden_size;
        T *dst = out + i * hidden_size;
        std::memcpy(dst, src, hidden_size * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t type, size_t seq_len, size_t hidden_size) {
    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), idx_ptr,
                         reinterpret_cast<const float *>(weight), seq_len, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), idx_ptr,
                         reinterpret_cast<const llaisys::bf16_t *>(weight), seq_len, hidden_size);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), idx_ptr,
                         reinterpret_cast<const llaisys::fp16_t *>(weight), seq_len, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
