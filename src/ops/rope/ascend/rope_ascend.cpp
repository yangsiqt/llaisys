#include "rope_ascend.hpp"
#include "../../../device/ascend/aclnn_utils.hpp"
#include "../../../utils/types.hpp"
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_cos.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_pow.h>
#include <aclnnop/aclnn_sin.h>
#include <aclnnop/aclnn_sub.h>
namespace llaisys::ops::ascend {
namespace {
template <typename GetWorkspace, typename Launch>
void unary(const device::ascend::AclTensorDesc &in, const device::ascend::AclTensorDesc &out,
           const char *workspace_name, const char *launch_name,
           GetWorkspace get_workspace, Launch launch) {
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    device::ascend::checkAclnn(get_workspace(in.get(), out.get(), &ws, &ex), workspace_name);
    device::ascend::runAclnn(ws, ex, launch_name, launch);
}

void cast(const device::ascend::AclTensorDesc &in, llaisysDataType_t dtype,
          const device::ascend::AclTensorDesc &out) {
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    device::ascend::checkAclnn(aclnnCastGetWorkspaceSize(in.get(), device::ascend::toAclDataType(dtype),
                                                         out.get(), &ws, &ex),
                               "aclnnCastGetWorkspaceSize(RoPE)");
    device::ascend::runAclnn(ws, ex, "aclnnCast(RoPE)", aclnnCast);
}

void powScalarTensor(float base, const device::ascend::AclTensorDesc &exponent,
                     const device::ascend::AclTensorDesc &out) {
    aclScalar *scalar = aclCreateScalar(&base, ACL_FLOAT);
    if (!scalar) throw std::runtime_error("aclCreateScalar(RoPE theta) returned null");
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    try {
        device::ascend::checkAclnn(aclnnPowScalarTensorGetWorkspaceSize(scalar, exponent.get(), out.get(), &ws, &ex),
                                   "aclnnPowScalarTensorGetWorkspaceSize(RoPE)");
        device::ascend::runAclnn(ws, ex, "aclnnPowScalarTensor(RoPE)", aclnnPowScalarTensor);
    } catch (...) {
        (void)aclDestroyScalar(scalar);
        throw;
    }
    device::ascend::checkAcl(aclDestroyScalar(scalar), "aclDestroyScalar(RoPE theta)");
}

void div(const device::ascend::AclTensorDesc &a, const device::ascend::AclTensorDesc &b,
         const device::ascend::AclTensorDesc &out) {
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    device::ascend::checkAclnn(aclnnDivGetWorkspaceSize(a.get(), b.get(), out.get(), &ws, &ex),
                               "aclnnDivGetWorkspaceSize(RoPE)");
    device::ascend::runAclnn(ws, ex, "aclnnDiv(RoPE)", aclnnDiv);
}

void mul(const device::ascend::AclTensorDesc &a, const device::ascend::AclTensorDesc &b,
         const device::ascend::AclTensorDesc &out) {
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    device::ascend::checkAclnn(aclnnMulGetWorkspaceSize(a.get(), b.get(), out.get(), &ws, &ex),
                               "aclnnMulGetWorkspaceSize(RoPE)");
    device::ascend::runAclnn(ws, ex, "aclnnMul(RoPE)", aclnnMul);
}

void copy(const device::ascend::AclTensorDesc &dst, const device::ascend::AclTensorDesc &src) {
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    device::ascend::checkAclnn(aclnnInplaceCopyGetWorkspaceSize(dst.get(), src.get(), &ws, &ex),
                               "aclnnInplaceCopyGetWorkspaceSize(RoPE)");
    device::ascend::runAclnn(ws, ex, "aclnnInplaceCopy(RoPE)", aclnnInplaceCopy);
}

template <typename GetWorkspace, typename Launch>
void addOrSub(const device::ascend::AclTensorDesc &a, const device::ascend::AclTensorDesc &b,
              const device::ascend::AclTensorDesc &out, const char *workspace_name,
              const char *launch_name, GetWorkspace get_workspace, Launch launch) {
    float one = 1.0F;
    aclScalar *alpha = aclCreateScalar(&one, ACL_FLOAT);
    if (!alpha) throw std::runtime_error("aclCreateScalar(RoPE alpha) returned null");
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    try {
        device::ascend::checkAclnn(get_workspace(a.get(), b.get(), alpha, out.get(), &ws, &ex), workspace_name);
        device::ascend::runAclnn(ws, ex, launch_name, launch);
    } catch (...) {
        (void)aclDestroyScalar(alpha);
        throw;
    }
    device::ascend::checkAcl(aclDestroyScalar(alpha), "aclDestroyScalar(RoPE alpha)");
}
}
void rope(void *out, const void *in, const int64_t *pos_ids, float theta, llaisysDataType_t dtype, size_t seq_len, size_t n_heads, size_t head_dim) {
    const size_t half = head_dim / 2;
    const size_t element_size = utils::dsize(dtype);
    if (dtype != LLAISYS_DTYPE_F32 && dtype != LLAISYS_DTYPE_F16 && dtype != LLAISYS_DTYPE_BF16)
        throw std::runtime_error("unsupported dtype for Ascend RoPE");
    size_t trig_bytes = seq_len * half * element_size;
    const size_t float_bytes = seq_len * half * sizeof(float);
    std::vector<float> exponents(half);
    for (size_t j = 0; j < half; ++j)
        exponents[j] = (2.0F * static_cast<float>(j)) / static_cast<float>(head_dim);
    device::ascend::DeviceBuffer positions_float(seq_len * sizeof(float));
    device::ascend::DeviceBuffer exponent_dev(half * sizeof(float)), denominator(half * sizeof(float));
    device::ascend::DeviceBuffer angle_dev(float_bytes), cos_float(float_bytes), sin_float(float_bytes);
    device::ascend::checkAcl(aclrtMemcpy(exponent_dev.data(), half * sizeof(float), exponents.data(),
                                         half * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE),
                             "aclrtMemcpy(RoPE exponents)");
    std::vector<int64_t> angle_shape{static_cast<int64_t>(seq_len), 1, static_cast<int64_t>(half)};
    std::vector<int64_t> position_shape{static_cast<int64_t>(seq_len), 1, 1};
    std::vector<int64_t> exponent_shape{1, 1, static_cast<int64_t>(half)};
    device::ascend::AclTensorDesc positions_desc(const_cast<int64_t *>(pos_ids), LLAISYS_DTYPE_I64, position_shape);
    device::ascend::AclTensorDesc positions_float_desc(positions_float.data(), LLAISYS_DTYPE_F32, position_shape);
    device::ascend::AclTensorDesc exponent_desc(exponent_dev.data(), LLAISYS_DTYPE_F32, exponent_shape);
    device::ascend::AclTensorDesc denominator_desc(denominator.data(), LLAISYS_DTYPE_F32, exponent_shape);
    device::ascend::AclTensorDesc angle_desc(angle_dev.data(), LLAISYS_DTYPE_F32, angle_shape);
    device::ascend::AclTensorDesc cos_float_desc(cos_float.data(), LLAISYS_DTYPE_F32, angle_shape);
    device::ascend::AclTensorDesc sin_float_desc(sin_float.data(), LLAISYS_DTYPE_F32, angle_shape);
    cast(positions_desc, LLAISYS_DTYPE_F32, positions_float_desc);
    powScalarTensor(theta, exponent_desc, denominator_desc);
    div(positions_float_desc, denominator_desc, angle_desc);
    unary(angle_desc, cos_float_desc, "aclnnCosGetWorkspaceSize(RoPE)", "aclnnCos(RoPE)",
          aclnnCosGetWorkspaceSize, aclnnCos);
    unary(angle_desc, sin_float_desc, "aclnnSinGetWorkspaceSize(RoPE)", "aclnnSin(RoPE)",
          aclnnSinGetWorkspaceSize, aclnnSin);
    device::ascend::DeviceBuffer cos_cast(dtype == LLAISYS_DTYPE_F32 ? 0 : trig_bytes);
    device::ascend::DeviceBuffer sin_cast(dtype == LLAISYS_DTYPE_F32 ? 0 : trig_bytes);
    void *cos_data = cos_float.data();
    void *sin_data = sin_float.data();
    if (dtype != LLAISYS_DTYPE_F32) {
        device::ascend::AclTensorDesc cos_cast_desc(cos_cast.data(), dtype, angle_shape);
        device::ascend::AclTensorDesc sin_cast_desc(sin_cast.data(), dtype, angle_shape);
        cast(cos_float_desc, dtype, cos_cast_desc);
        cast(sin_float_desc, dtype, sin_cast_desc);
        cos_data = cos_cast.data();
        sin_data = sin_cast.data();
    }
    std::vector<int64_t> hshape{static_cast<int64_t>(seq_len), static_cast<int64_t>(n_heads), static_cast<int64_t>(half)};
    std::vector<int64_t> xstorage{static_cast<int64_t>(seq_len), static_cast<int64_t>(n_heads), static_cast<int64_t>(head_dim)};
    std::vector<int64_t> xstrides{static_cast<int64_t>(n_heads * head_dim), static_cast<int64_t>(head_dim), 1};
    std::vector<int64_t> tshape{static_cast<int64_t>(seq_len), 1, static_cast<int64_t>(half)};
    auto *input = static_cast<const std::byte *>(in);
    device::ascend::AclTensorDesc xa(const_cast<std::byte *>(input), dtype, hshape, xstrides, xstorage);
    device::ascend::AclTensorDesc xb(const_cast<std::byte *>(input + half * element_size), dtype, hshape, xstrides, xstorage);
    device::ascend::AclTensorDesc c(cos_data, dtype, tshape), s(sin_data, dtype, tshape);

    const size_t half_numel = seq_len * n_heads * half;
    const size_t half_bytes = half_numel * element_size;
    device::ascend::DeviceBuffer ac_buf(half_bytes), bs_buf(half_bytes), bc_buf(half_bytes), as_buf(half_bytes);
    device::ascend::AclTensorDesc ac(ac_buf.data(), dtype, hshape), bs(bs_buf.data(), dtype, hshape);
    device::ascend::AclTensorDesc bc(bc_buf.data(), dtype, hshape), as(as_buf.data(), dtype, hshape);
    mul(xa, c, ac);
    mul(xb, s, bs);
    addOrSub(ac, bs, ac, "aclnnSubGetWorkspaceSize(RoPE)", "aclnnSub(RoPE)",
             aclnnSubGetWorkspaceSize, aclnnSub);
    mul(xb, c, bc);
    mul(xa, s, as);
    addOrSub(bc, as, bc, "aclnnAddGetWorkspaceSize(RoPE)", "aclnnAdd(RoPE)",
             aclnnAddGetWorkspaceSize, aclnnAdd);

    // Copy the two contiguous half-results into strided views of the
    // interleaved per-head layout. Both results are materialized first, so
    // out may alias in.
    auto *output = static_cast<std::byte *>(out);
    device::ascend::AclTensorDesc ya(output, dtype, hshape, xstrides, xstorage);
    device::ascend::AclTensorDesc yb(output + half * element_size, dtype, hshape, xstrides, xstorage);
    copy(ya, ac);
    copy(yb, bc);
}
}
