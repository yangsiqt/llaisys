#include "op.hpp"
#include "cpu/add_cpu.hpp"
#include "../../utils.hpp"
#include "../../core/llaisys_core.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/add_nvidia.hpp"
#endif

namespace llaisys {
namespace ops {

void add(const tensor_t& out, const tensor_t& a, const tensor_t& b) {
    CHECK_ARGUMENT(out->shape() == a->shape(), "Output and input a must have the same shape");
    CHECK_ARGUMENT(out->shape() == b->shape(), "Output and input b must have the same shape");
    CHECK_ARGUMENT(out->dtype() == a->dtype() && out->dtype() == b->dtype(), 
                   "All tensors must have the same dtype");
    CHECK_ARGUMENT(out->deviceType() == a->deviceType() && out->deviceType() == b->deviceType(),
                   "All tensors must be on the same device");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::add_cpu(out, a, b);
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::add_cpu(out, a, b);
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::add(out->data(), a->data(), b->data(), out->dtype(), out->numel());
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace ops
} // namespace llaisys
