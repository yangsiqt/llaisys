#ifndef LLAISYS_OPS_ADD_CPU_HPP
#define LLAISYS_OPS_ADD_CPU_HPP

#include "../../../tensor/tensor.hpp"

namespace llaisys {
namespace ops {
namespace cpu {

void add_cpu(const tensor_t& out, const tensor_t& a, const tensor_t& b);

} // namespace cpu
} // namespace ops
} // namespace llaisys

#endif // LLAISYS_OPS_ADD_CPU_HPP
