#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rearrange_nvidia.hpp"
#endif

#include <cstring>

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(out->isContiguous(), "out must be contiguous");

    // If input is already contiguous and has same strides, just copy
    if (in->isContiguous() && in->strides() == out->strides()) {
        size_t size = in->numel() * in->elementSize();
        if (in->deviceType() == LLAISYS_DEVICE_CPU) {
            std::memcpy(out->data(), in->data(), size);
        } else {
            llaisys::core::context().setDevice(in->deviceType(), in->deviceId());
            llaisys::core::context().runtime().api()->memcpy_sync(
                out->data(), in->data(), size, LLAISYS_MEMCPY_D2D);
        }
        return;
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(out->data(), in->data(), 
                             in->shape(), out->strides(), in->strides(),
                             out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(out->data(), in->data(), 
                             in->shape(), out->strides(), in->strides(),
                             out->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rearrange(out->data(), in->data(), in->shape(), out->strides(), in->strides(), out->dtype());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
