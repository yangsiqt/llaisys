#include "swiglu_ascend.hpp"
#include "../../../device/ascend/aclnn_utils.hpp"
#include "../../../utils/types.hpp"
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_silu.h>
namespace llaisys::ops::ascend {
void swiglu(void *out, const void *gate, const void *up, llaisysDataType_t dtype, size_t numel) {
    std::vector<int64_t> shape{static_cast<int64_t>(numel)};
    device::ascend::DeviceBuffer silu_buf(numel * utils::dsize(dtype));
    device::ascend::AclTensorDesc g(const_cast<void *>(gate), dtype, shape), u(const_cast<void *>(up), dtype, shape), s(silu_buf.data(), dtype, shape), o(out, dtype, shape);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    device::ascend::checkAclnn(aclnnSiluGetWorkspaceSize(g.get(), s.get(), &ws, &ex), "aclnnSiluGetWorkspaceSize");
    device::ascend::runAclnn(ws, ex, "aclnnSilu", aclnnSilu);
    ws = 0; ex = nullptr;
    device::ascend::checkAclnn(aclnnMulGetWorkspaceSize(s.get(), u.get(), o.get(), &ws, &ex), "aclnnMulGetWorkspaceSize");
    device::ascend::runAclnn(ws, ex, "aclnnMul", aclnnMul);
}
}
