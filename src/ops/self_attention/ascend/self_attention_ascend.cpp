#include "self_attention_ascend.hpp"
#include "../../../device/ascend/aclnn_utils.hpp"
#include "../../../utils/types.hpp"
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_softmax.h>
#include <limits>
namespace llaisys::ops::ascend {
namespace {
template <typename T> void fillMask(std::vector<std::byte> &data, size_t qlen, size_t kvlen) {
    auto *mask = reinterpret_cast<T *>(data.data());
    size_t offset = kvlen - qlen;
    for (size_t qi = 0; qi < qlen; ++qi)
        for (size_t ki = 0; ki < kvlen; ++ki)
            mask[qi * kvlen + ki] = utils::cast<T>(ki <= offset + qi ? 0.0f : -std::numeric_limits<float>::infinity());
}
}
void self_attention(void *out, const void *q, const void *k, const void *v, float scale,
                    llaisysDataType_t dtype, size_t qlen, size_t kvlen,
                    size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    size_t group = n_heads / n_kv_heads;
    std::vector<int64_t> q_view{static_cast<int64_t>(n_kv_heads), static_cast<int64_t>(group), static_cast<int64_t>(qlen), static_cast<int64_t>(head_dim)};
    std::vector<int64_t> q_stride{static_cast<int64_t>(group * head_dim), static_cast<int64_t>(head_dim), static_cast<int64_t>(n_heads * head_dim), 1};
    std::vector<int64_t> q_storage{static_cast<int64_t>(qlen), static_cast<int64_t>(n_heads), static_cast<int64_t>(head_dim)};
    std::vector<int64_t> k_view{static_cast<int64_t>(n_kv_heads), 1, static_cast<int64_t>(head_dim), static_cast<int64_t>(kvlen)};
    std::vector<int64_t> k_stride{static_cast<int64_t>(head_dim), 0, 1, static_cast<int64_t>(n_kv_heads * head_dim)};
    std::vector<int64_t> kv_storage{static_cast<int64_t>(kvlen), static_cast<int64_t>(n_kv_heads), static_cast<int64_t>(head_dim)};
    std::vector<int64_t> v_view{static_cast<int64_t>(n_kv_heads), 1, static_cast<int64_t>(kvlen), static_cast<int64_t>(head_dim)};
    std::vector<int64_t> v_stride{static_cast<int64_t>(head_dim), 0, static_cast<int64_t>(n_kv_heads * head_dim), 1};
    std::vector<int64_t> score_shape{static_cast<int64_t>(n_kv_heads), static_cast<int64_t>(group), static_cast<int64_t>(qlen), static_cast<int64_t>(kvlen)};
    size_t score_elems = n_heads * qlen * kvlen;
    size_t elem_size = utils::dsize(dtype);
    device::ascend::DeviceBuffer scores(score_elems * elem_size), probs(score_elems * elem_size);
    device::ascend::AclTensorDesc qd(const_cast<void *>(q), dtype, q_view, q_stride, q_storage);
    device::ascend::AclTensorDesc kd(const_cast<void *>(k), dtype, k_view, k_stride, kv_storage);
    device::ascend::AclTensorDesc sd(scores.data(), dtype, score_shape);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    device::ascend::checkAclnn(aclnnMatmulGetWorkspaceSize(qd.get(), kd.get(), sd.get(), 0, &ws, &ex), "aclnnMatmulGetWorkspaceSize(qk)");
    device::ascend::runAclnn(ws, ex, "aclnnMatmul(qk)", aclnnMatmul);

    aclScalar *scale_scalar = aclCreateScalar(&scale, ACL_FLOAT);
    if (!scale_scalar) throw std::runtime_error("aclCreateScalar(scale) returned null");
    ws = 0; ex = nullptr;
    auto st = aclnnInplaceMulsGetWorkspaceSize(sd.get(), scale_scalar, &ws, &ex);
    device::ascend::checkAclnn(st, "aclnnInplaceMulsGetWorkspaceSize");
    device::ascend::runAclnn(ws, ex, "aclnnInplaceMuls", aclnnInplaceMuls);
    device::ascend::checkAcl(aclDestroyScalar(scale_scalar), "aclDestroyScalar(scale)");

    size_t mask_elems = qlen * kvlen;
    std::vector<std::byte> mask_host(mask_elems * elem_size);
    if (dtype == LLAISYS_DTYPE_F32) fillMask<float>(mask_host, qlen, kvlen);
    else if (dtype == LLAISYS_DTYPE_F16) fillMask<fp16_t>(mask_host, qlen, kvlen);
    else if (dtype == LLAISYS_DTYPE_BF16) fillMask<bf16_t>(mask_host, qlen, kvlen);
    else throw std::runtime_error("unsupported dtype for Ascend self_attention");
    device::ascend::DeviceBuffer mask_dev(mask_host.size());
    device::ascend::checkAcl(aclrtMemcpy(mask_dev.data(), mask_host.size(), mask_host.data(), mask_host.size(), ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(attention mask)");
    device::ascend::AclTensorDesc md(mask_dev.data(), dtype, {1, 1, static_cast<int64_t>(qlen), static_cast<int64_t>(kvlen)});
    float one = 1.0f; aclScalar *alpha = aclCreateScalar(&one, ACL_FLOAT);
    if (!alpha) throw std::runtime_error("aclCreateScalar(alpha) returned null");
    ws = 0; ex = nullptr;
    st = aclnnInplaceAddGetWorkspaceSize(sd.get(), md.get(), alpha, &ws, &ex);
    device::ascend::checkAclnn(st, "aclnnInplaceAddGetWorkspaceSize(mask)");
    device::ascend::runAclnn(ws, ex, "aclnnInplaceAdd(mask)", aclnnInplaceAdd);
    device::ascend::checkAcl(aclDestroyScalar(alpha), "aclDestroyScalar(mask alpha)");

    device::ascend::AclTensorDesc pd(probs.data(), dtype, score_shape);
    ws = 0; ex = nullptr;
    device::ascend::checkAclnn(aclnnSoftmaxGetWorkspaceSize(sd.get(), -1, pd.get(), &ws, &ex), "aclnnSoftmaxGetWorkspaceSize");
    device::ascend::runAclnn(ws, ex, "aclnnSoftmax", aclnnSoftmax);

    device::ascend::AclTensorDesc vd(const_cast<void *>(v), dtype, v_view, v_stride, kv_storage);
    device::ascend::AclTensorDesc od(out, dtype, q_view, q_stride, q_storage);
    ws = 0; ex = nullptr;
    device::ascend::checkAclnn(aclnnMatmulGetWorkspaceSize(pd.get(), vd.get(), od.get(), 0, &ws, &ex), "aclnnMatmulGetWorkspaceSize(pv)");
    device::ascend::runAclnn(ws, ex, "aclnnMatmul(pv)", aclnnMatmul);
}
}
