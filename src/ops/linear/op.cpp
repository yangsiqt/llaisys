#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // Check device consistency
    if (bias) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    }
    
    CHECK_ARGUMENT(in->ndim() == 2, "in must be 2D");
    CHECK_ARGUMENT(weight->ndim() == 2, "weight must be 2D");
    CHECK_ARGUMENT(out->ndim() == 2, "out must be 2D");
    CHECK_ARGUMENT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
                   "All tensors must be contiguous");
    
    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    
    CHECK_ARGUMENT(weight->shape()[1] == in_features, "weight shape[1] must match in_features");
    CHECK_ARGUMENT(out->shape()[0] == batch_size, "out shape[0] must match batch_size");
    CHECK_ARGUMENT(out->shape()[1] == out_features, "out shape[1] must match out_features");
    
    if (bias) {
        CHECK_ARGUMENT(bias->ndim() == 1, "bias must be 1D");
        CHECK_ARGUMENT(bias->shape()[0] == out_features, "bias length must match out_features");
        CHECK_ARGUMENT(bias->isContiguous(), "bias must be contiguous");
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          bias ? bias->data() : nullptr,
                          out->dtype(), batch_size, in_features, out_features);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          bias ? bias->data() : nullptr,
                          out->dtype(), batch_size, in_features, out_features);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
