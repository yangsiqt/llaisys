#include "argmax_ascend.hpp"
#include "../../../device/ascend/aclnn_utils.hpp"
#include "../../../utils/types.hpp"
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_max_dim.h>
namespace llaisys::ops::ascend {
namespace {
void castTensor(void *out, llaisysDataType_t out_dtype, const void *in, llaisysDataType_t in_dtype, const std::vector<int64_t> &shape) {
    device::ascend::AclTensorDesc src(const_cast<void *>(in), in_dtype, shape), dst(out, out_dtype, shape);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    device::ascend::checkAclnn(aclnnCastGetWorkspaceSize(src.get(), device::ascend::toAclDataType(out_dtype), dst.get(), &ws, &ex), "aclnnCastGetWorkspaceSize");
    device::ascend::runAclnn(ws, ex, "aclnnCast", aclnnCast);
}
}
void argmax(void *max_idx, void *max_val, const void *vals, llaisysDataType_t dtype, size_t numel) {
    std::vector<int64_t> shape{static_cast<int64_t>(numel)}, one{1};
    const void *max_input = vals;
    llaisysDataType_t compute_dtype = dtype;
    device::ascend::DeviceBuffer float_input(dtype == LLAISYS_DTYPE_BF16 ? numel * sizeof(float) : 0);
    device::ascend::DeviceBuffer float_value(dtype == LLAISYS_DTYPE_BF16 ? sizeof(float) : 0);
    if (dtype == LLAISYS_DTYPE_BF16) {
        castTensor(float_input.data(), LLAISYS_DTYPE_F32, vals, dtype, shape);
        max_input = float_input.data(); compute_dtype = LLAISYS_DTYPE_F32;
    }
    void *value_output = dtype == LLAISYS_DTYPE_BF16 ? float_value.data() : max_val;
    device::ascend::AclTensorDesc src(const_cast<void *>(max_input), compute_dtype, shape);
    device::ascend::AclTensorDesc value(value_output, compute_dtype, one);
    device::ascend::AclTensorDesc index(max_idx, LLAISYS_DTYPE_I64, one);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    device::ascend::checkAclnn(aclnnMaxDimGetWorkspaceSize(src.get(), 0, true, value.get(), index.get(), &ws, &ex), "aclnnMaxDimGetWorkspaceSize");
    device::ascend::runAclnn(ws, ex, "aclnnMaxDim", aclnnMaxDim);
    if (dtype == LLAISYS_DTYPE_BF16) castTensor(max_val, dtype, float_value.data(), LLAISYS_DTYPE_F32, one);
}
}
