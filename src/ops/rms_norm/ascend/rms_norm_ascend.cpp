#include "rms_norm_ascend.hpp"
#include "../../../device/ascend/aclnn_utils.hpp"
#include <aclnnop/aclnn_rms_norm.h>
namespace llaisys::ops::ascend {
void rms_norm(void *out, const void *in, const void *weight, float eps, llaisysDataType_t dtype, size_t batch_size, size_t hidden_size) {
    device::ascend::AclTensorDesc x(const_cast<void *>(in), dtype, {static_cast<int64_t>(batch_size), static_cast<int64_t>(hidden_size)});
    device::ascend::AclTensorDesc w(const_cast<void *>(weight), dtype, {static_cast<int64_t>(hidden_size)});
    device::ascend::AclTensorDesc y(out, dtype, {static_cast<int64_t>(batch_size), static_cast<int64_t>(hidden_size)});
    device::ascend::DeviceBuffer rstd_buf(batch_size * sizeof(float));
    device::ascend::AclTensorDesc rstd(rstd_buf.data(), LLAISYS_DTYPE_F32, {static_cast<int64_t>(batch_size), 1});
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    device::ascend::checkAclnn(aclnnRmsNormGetWorkspaceSize(x.get(), w.get(), eps, y.get(), rstd.get(), &ws, &ex), "aclnnRmsNormGetWorkspaceSize");
    device::ascend::runAclnn(ws, ex, "aclnnRmsNorm", aclnnRmsNorm);
}
}
