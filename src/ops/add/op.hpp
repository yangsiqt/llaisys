#ifndef LLAISYS_OPS_ADD_HPP
#define LLAISYS_OPS_ADD_HPP

#include "../../tensor/tensor.hpp"

namespace llaisys {
namespace ops {

// Element-wise addition: out = a + b
void add(const tensor_t& out, const tensor_t& a, const tensor_t& b);

} // namespace ops
} // namespace llaisys

#endif // LLAISYS_OPS_ADD_HPP
