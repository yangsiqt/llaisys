#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.hpp"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "pos_ids must be int64");
    CHECK_ARGUMENT(in->ndim() == 3, "in must be 3D [seq_len, n_heads, head_dim]");
    CHECK_ARGUMENT(out->ndim() == 3, "out must be 3D");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "pos_ids must be 1D");
    CHECK_ARGUMENT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
                   "All tensors must be contiguous");
    
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];
    
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(pos_ids->shape()[0] == seq_len, "pos_ids length must match seq_len");
    CHECK_ARGUMENT(head_dim % 2 == 0, "head_dim must be even for RoPE");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta,
                        out->dtype(), seq_len, n_heads, head_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta,
                        out->dtype(), seq_len, n_heads, head_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), seq_len, n_heads, head_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
