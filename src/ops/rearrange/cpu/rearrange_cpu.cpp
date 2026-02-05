#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>
#include <vector>

// Rearrange: copy data from one tensor to another with same shape but different strides
// This is useful for making non-contiguous tensors contiguous

template <typename T>
void rearrange_recursive(T *out, const T *in, 
                        const std::vector<size_t> &shape,
                        const std::vector<ptrdiff_t> &out_strides,
                        const std::vector<ptrdiff_t> &in_strides,
                        size_t dim, size_t out_offset, size_t in_offset) {
    if (dim == shape.size() - 1) {
        // Last dimension: copy elements
        for (size_t i = 0; i < shape[dim]; i++) {
            out[out_offset + i * out_strides[dim]] = in[in_offset + i * in_strides[dim]];
        }
    } else {
        // Recursive case: iterate over current dimension
        for (size_t i = 0; i < shape[dim]; i++) {
            rearrange_recursive(out, in, shape, out_strides, in_strides, 
                              dim + 1, 
                              out_offset + i * out_strides[dim],
                              in_offset + i * in_strides[dim]);
        }
    }
}

template <typename T>
void rearrange_(T *out, const T *in, 
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides) {
    
    if (shape.empty()) {
        // Scalar case
        out[0] = in[0];
        return;
    }
    
    rearrange_recursive(out, in, shape, out_strides, in_strides, 0, 0, 0);
}

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, 
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides,
               llaisysDataType_t type) {
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rearrange_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         shape, out_strides, in_strides);
    case LLAISYS_DTYPE_BF16:
        return rearrange_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         shape, out_strides, in_strides);
    case LLAISYS_DTYPE_F16:
        return rearrange_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         shape, out_strides, in_strides);
    case LLAISYS_DTYPE_I64:
        return rearrange_(reinterpret_cast<int64_t *>(out),
                         reinterpret_cast<const int64_t *>(in),
                         shape, out_strides, in_strides);
    case LLAISYS_DTYPE_I32:
        return rearrange_(reinterpret_cast<int32_t *>(out),
                         reinterpret_cast<const int32_t *>(in),
                         shape, out_strides, in_strides);
    case LLAISYS_DTYPE_I16:
        return rearrange_(reinterpret_cast<int16_t *>(out),
                         reinterpret_cast<const int16_t *>(in),
                         shape, out_strides, in_strides);
    case LLAISYS_DTYPE_I8:
        return rearrange_(reinterpret_cast<int8_t *>(out),
                         reinterpret_cast<const int8_t *>(in),
                         shape, out_strides, in_strides);
    case LLAISYS_DTYPE_U64:
        return rearrange_(reinterpret_cast<uint64_t *>(out),
                         reinterpret_cast<const uint64_t *>(in),
                         shape, out_strides, in_strides);
    case LLAISYS_DTYPE_U32:
        return rearrange_(reinterpret_cast<uint32_t *>(out),
                         reinterpret_cast<const uint32_t *>(in),
                         shape, out_strides, in_strides);
    case LLAISYS_DTYPE_U16:
        return rearrange_(reinterpret_cast<uint16_t *>(out),
                         reinterpret_cast<const uint16_t *>(in),
                         shape, out_strides, in_strides);
    case LLAISYS_DTYPE_U8:
        return rearrange_(reinterpret_cast<uint8_t *>(out),
                         reinterpret_cast<const uint8_t *>(in),
                         shape, out_strides, in_strides);
    case LLAISYS_DTYPE_BOOL:
        return rearrange_(reinterpret_cast<bool *>(out),
                         reinterpret_cast<const bool *>(in),
                         shape, out_strides, in_strides);
    case LLAISYS_DTYPE_F64:
        return rearrange_(reinterpret_cast<double *>(out),
                         reinterpret_cast<const double *>(in),
                         shape, out_strides, in_strides);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu

