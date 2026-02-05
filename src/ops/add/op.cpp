#include "op.hpp"
#include "cpu/add_cpu.hpp"
#include "../../utils.hpp"

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
    } else {
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace ops
} // namespace llaisys
