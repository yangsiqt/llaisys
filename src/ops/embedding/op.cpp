#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.hpp"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "index must be int64");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(index->ndim() == 1, "index must be 1D");
    CHECK_ARGUMENT(weight->ndim() == 2, "weight must be 2D");
    CHECK_ARGUMENT(out->ndim() == 2, "out must be 2D");
    CHECK_ARGUMENT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
                   "All tensors must be contiguous");
    
    size_t seq_len = index->shape()[0];
    size_t hidden_size = weight->shape()[1];
    
    CHECK_ARGUMENT(out->shape()[0] == seq_len, "out shape[0] must match index length");
    CHECK_ARGUMENT(out->shape()[1] == hidden_size, "out shape[1] must match weight hidden_size");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                            out->dtype(), seq_len, hidden_size);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                            out->dtype(), seq_len, hidden_size);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), out->dtype(), seq_len, hidden_size);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
