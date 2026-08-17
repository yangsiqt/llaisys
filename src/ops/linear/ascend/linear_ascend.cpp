#include "linear_ascend.hpp"
#include "../../../device/ascend/aclnn_utils.hpp"
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_matmul.h>
namespace llaisys::ops::ascend {
void linear(void *out, const void *in, const void *weight, const void *bias, llaisysDataType_t dtype, size_t batch_size, size_t in_features, size_t out_features) {
    std::vector<int64_t> in_shape{static_cast<int64_t>(batch_size), static_cast<int64_t>(in_features)};
    std::vector<int64_t> out_shape{static_cast<int64_t>(batch_size), static_cast<int64_t>(out_features)};
    device::ascend::AclTensorDesc x(const_cast<void *>(in), dtype, in_shape);
    device::ascend::AclTensorDesc w(const_cast<void *>(weight), dtype,
        {static_cast<int64_t>(in_features), static_cast<int64_t>(out_features)},
        {1, static_cast<int64_t>(in_features)},
        {static_cast<int64_t>(out_features), static_cast<int64_t>(in_features)});
    device::ascend::AclTensorDesc y(out, dtype, out_shape);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    device::ascend::checkAclnn(aclnnMatmulGetWorkspaceSize(x.get(), w.get(), y.get(), 0, &ws, &ex), "aclnnMatmulGetWorkspaceSize");
    device::ascend::runAclnn(ws, ex, "aclnnMatmul", aclnnMatmul);
    if (bias) {
        device::ascend::AclTensorDesc b(const_cast<void *>(bias), dtype, {static_cast<int64_t>(out_features)});
        float one = 1.0f; aclScalar *alpha = aclCreateScalar(&one, ACL_FLOAT);
        if (!alpha) throw std::runtime_error("aclCreateScalar(alpha) returned null");
        ws = 0; ex = nullptr;
        auto st = aclnnInplaceAddGetWorkspaceSize(y.get(), b.get(), alpha, &ws, &ex);
        device::ascend::checkAclnn(st, "aclnnInplaceAddGetWorkspaceSize");
        device::ascend::runAclnn(ws, ex, "aclnnInplaceAdd", aclnnInplaceAdd);
        device::ascend::checkAcl(aclDestroyScalar(alpha), "aclDestroyScalar(linear alpha)");
    }
}
}
