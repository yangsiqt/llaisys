#include "embedding_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

template<typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                 size_t seq_len, size_t hidden_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * hidden_size) {
        size_t s = idx / hidden_size, h = idx % hidden_size;
        out[s * hidden_size + h] = weight[index[s] * hidden_size + h];
    }
}

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t seq_len, size_t hidden_size) {
    size_t total = seq_len * hidden_size;
    int threads = 256, blocks = (total + threads - 1) / threads;
    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<float><<<blocks,threads>>>(reinterpret_cast<float*>(out),reinterpret_cast<const int64_t*>(index),reinterpret_cast<const float*>(weight),seq_len,hidden_size); break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<__nv_bfloat16><<<blocks,threads>>>(reinterpret_cast<__nv_bfloat16*>(out),reinterpret_cast<const int64_t*>(index),reinterpret_cast<const __nv_bfloat16*>(weight),seq_len,hidden_size); break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<__half><<<blocks,threads>>>(reinterpret_cast<__half*>(out),reinterpret_cast<const int64_t*>(index),reinterpret_cast<const __half*>(weight),seq_len,hidden_size); break;
    default: throw std::runtime_error("Unsupported dtype for CUDA embedding");
    }
}
} // namespace llaisys::ops::nvidia
