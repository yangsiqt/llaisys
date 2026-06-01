#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.hpp"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "max_idx must be int64");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    CHECK_ARGUMENT(vals->isContiguous(), "vals must be contiguous");
    CHECK_ARGUMENT(max_idx->numel() == 1, "max_idx must have 1 element");
    CHECK_ARGUMENT(max_val->numel() == 1, "max_val must have 1 element");

    // always support cpu calculation
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void argmax_batch(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "max_idx must be int64");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    CHECK_ARGUMENT(vals->isContiguous(), "vals must be contiguous");
    CHECK_ARGUMENT(vals->ndim() == 2, "vals must be 2D [batch, width]");
    CHECK_ARGUMENT(max_idx->ndim() == 1 && max_val->ndim() == 1, "max_idx/max_val must be 1D");
    CHECK_ARGUMENT(max_idx->shape()[0] == vals->shape()[0], "max_idx batch mismatch");
    CHECK_ARGUMENT(max_val->shape()[0] == vals->shape()[0], "max_val batch mismatch");

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax_batch(max_idx->data(), max_val->data(), vals->data(), vals->dtype(),
                                    vals->shape()[0], vals->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void argmax_batch_fast(tensor_t max_idx, tensor_t max_val, tensor_t vals,
                       tensor_t partial_idx, tensor_t partial_val) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals, partial_idx, partial_val);
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "max_idx must be int64");
    CHECK_ARGUMENT(partial_idx->dtype() == LLAISYS_DTYPE_I64, "partial_idx must be int64");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype(), partial_val->dtype());
    CHECK_ARGUMENT(vals->isContiguous() && partial_idx->isContiguous() && partial_val->isContiguous(),
                   "vals and partial buffers must be contiguous");
    CHECK_ARGUMENT(vals->ndim() == 2, "vals must be 2D [batch, width]");
    CHECK_ARGUMENT(max_idx->ndim() == 1 && max_val->ndim() == 1, "max_idx/max_val must be 1D");
    CHECK_ARGUMENT(partial_idx->ndim() == 2 && partial_val->ndim() == 2,
                   "partial buffers must be 2D [batch, parts]");
    CHECK_ARGUMENT(max_idx->shape()[0] == vals->shape()[0], "max_idx batch mismatch");
    CHECK_ARGUMENT(max_val->shape()[0] == vals->shape()[0], "max_val batch mismatch");
    CHECK_ARGUMENT(partial_idx->shape()[0] == vals->shape()[0], "partial_idx batch mismatch");
    CHECK_ARGUMENT(partial_val->shape()[0] == vals->shape()[0], "partial_val batch mismatch");
    CHECK_ARGUMENT(partial_idx->shape()[1] == partial_val->shape()[1], "partial buffer parts mismatch");
    CHECK_ARGUMENT(partial_idx->shape()[1] > 0, "partial parts must be > 0");

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax_batch_fast(max_idx->data(), max_val->data(), vals->data(),
                                         partial_idx->data(), partial_val->data(),
                                         vals->dtype(), vals->shape()[0], vals->shape()[1],
                                         partial_idx->shape()[1]);
#endif
    default:
        return argmax_batch(max_idx, max_val, vals);
    }
}
} // namespace llaisys::ops
