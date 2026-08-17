#include "rearrange_ascend.hpp"
#include "../../../device/ascend/aclnn_utils.hpp"
#include <aclnnop/aclnn_copy.h>
namespace llaisys::ops::ascend {
void rearrange(void *out, const void *in, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &out_strides, const std::vector<ptrdiff_t> &in_strides, llaisysDataType_t dtype) {
    std::vector<int64_t> dims, os, is;
    for (auto v : shape) dims.push_back(static_cast<int64_t>(v));
    for (auto v : out_strides) os.push_back(static_cast<int64_t>(v));
    for (auto v : in_strides) is.push_back(static_cast<int64_t>(v));
    device::ascend::AclTensorDesc src(const_cast<void *>(in), dtype, dims, is, dims);
    device::ascend::AclTensorDesc dst(out, dtype, dims, os, dims);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    device::ascend::checkAclnn(aclnnInplaceCopyGetWorkspaceSize(dst.get(), src.get(), &ws, &ex), "aclnnInplaceCopyGetWorkspaceSize");
    device::ascend::runAclnn(ws, ex, "aclnnInplaceCopy", aclnnInplaceCopy);
}
}
