#include "add_ascend.hpp"
#include "../../../device/ascend/aclnn_utils.hpp"
#include <aclnnop/aclnn_add.h>
namespace llaisys::ops::ascend {
void add(void *out, const void *a, const void *b, llaisysDataType_t dtype, size_t numel) {
    std::vector<int64_t> shape{static_cast<int64_t>(numel)};
    device::ascend::AclTensorDesc oa(out, dtype, shape), aa(const_cast<void *>(a), dtype, shape), ba(const_cast<void *>(b), dtype, shape);
    float one = 1.0f;
    aclScalar *alpha = aclCreateScalar(&one, ACL_FLOAT);
    if (!alpha) throw std::runtime_error("aclCreateScalar(alpha) returned null");
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    auto st = aclnnAddGetWorkspaceSize(aa.get(), ba.get(), alpha, oa.get(), &ws, &ex);
    device::ascend::checkAclnn(st, "aclnnAddGetWorkspaceSize");
    device::ascend::runAclnn(ws, ex, "aclnnAdd", aclnnAdd);
    device::ascend::checkAcl(aclDestroyScalar(alpha), "aclDestroyScalar(add alpha)");
}
}
