#include "rearrange_nvidia.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

#define MAX_DIMS 8
struct RearrangeParams {
    size_t shape[MAX_DIMS];
    ptrdiff_t out_strides[MAX_DIMS];
    ptrdiff_t in_strides[MAX_DIMS];
    int ndim;
    size_t numel;
};

template<typename T>
__global__ void rearrange_kernel(T *out, const T *in, RearrangeParams p) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.numel) return;
    size_t remaining = idx, in_off = 0, out_off = 0;
    for (int d = 0; d < p.ndim; d++) {
        size_t next = 1;
        for (int dd = d+1; dd < p.ndim; dd++) next *= p.shape[dd];
        size_t coord = remaining / next; remaining %= next;
        out_off += coord * p.out_strides[d];
        in_off  += coord * p.in_strides[d];
    }
    out[out_off] = in[in_off];
}

namespace llaisys::ops::nvidia {
void rearrange(std::byte *out, const std::byte *in,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides,
               llaisysDataType_t type) {
    if (shape.empty()) return;
    if ((int)shape.size() > MAX_DIMS) throw std::runtime_error("rearrange: too many dims");
    RearrangeParams p; p.ndim = shape.size(); p.numel = 1;
    for (int d = 0; d < p.ndim; d++) {
        p.shape[d] = shape[d]; p.out_strides[d] = out_strides[d]; p.in_strides[d] = in_strides[d];
        p.numel *= shape[d];
    }
    int t = 256, b = (p.numel + t - 1) / t;
    size_t esz = 0;
    switch (type) {
    case LLAISYS_DTYPE_F32: case LLAISYS_DTYPE_I32: case LLAISYS_DTYPE_U32: esz=4; break;
    case LLAISYS_DTYPE_F64: case LLAISYS_DTYPE_I64: case LLAISYS_DTYPE_U64: esz=8; break;
    case LLAISYS_DTYPE_F16: case LLAISYS_DTYPE_BF16: case LLAISYS_DTYPE_I16: case LLAISYS_DTYPE_U16: esz=2; break;
    default: esz=1;
    }
    switch (esz) {
    case 1: rearrange_kernel<uint8_t><<<b,t>>>((uint8_t*)out,(const uint8_t*)in,p); break;
    case 2: rearrange_kernel<uint16_t><<<b,t>>>((uint16_t*)out,(const uint16_t*)in,p); break;
    case 4: rearrange_kernel<uint32_t><<<b,t>>>((uint32_t*)out,(const uint32_t*)in,p); break;
    case 8: rearrange_kernel<uint64_t><<<b,t>>>((uint64_t*)out,(const uint64_t*)in,p); break;
    }
}
} // namespace llaisys::ops::nvidia
