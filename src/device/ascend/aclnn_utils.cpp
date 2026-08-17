#include "aclnn_utils.hpp"

#include "../../core/context/context.hpp"

#include <aclnn/aclnn_base.h>

#include <sstream>
#include <stdexcept>

namespace llaisys::device::ascend {
namespace {

class WorkspaceCache {
public:
    ~WorkspaceCache() {
        if (data_) (void)aclrtFree(data_);
    }

    void *acquire(size_t bytes) {
        if (bytes <= capacity_) return data_;
        // Executors launch asynchronously. Never release the old workspace
        // until all work using it on the LLAISYS stream has completed.
        checkAcl(aclrtSynchronizeStream(currentStream()), "aclrtSynchronizeStream(workspace grow)");
        if (data_) checkAcl(aclrtFree(data_), "aclrtFree(workspace grow)");
        data_ = nullptr;
        capacity_ = 0;
        checkAcl(aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(workspace)");
        capacity_ = bytes;
        return data_;
    }

private:
    void *data_ = nullptr;
    size_t capacity_ = 0;
};

WorkspaceCache &workspaceCache() {
    thread_local WorkspaceCache cache;
    return cache;
}

} // namespace

void checkAcl(aclError status, const char *call) {
    if (status != ACL_SUCCESS) {
        std::ostringstream os;
        os << call << " failed with ACL error " << status;
        throw std::runtime_error(os.str());
    }
}

void checkAclnn(aclnnStatus status, const char *call) {
    if (status != 0) {
        std::ostringstream os;
        os << call << " failed with ACLNN error " << status;
        throw std::runtime_error(os.str());
    }
}

aclDataType toAclDataType(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BOOL: return ACL_BOOL;
    case LLAISYS_DTYPE_I8: return ACL_INT8;
    case LLAISYS_DTYPE_I16: return ACL_INT16;
    case LLAISYS_DTYPE_I32: return ACL_INT32;
    case LLAISYS_DTYPE_I64: return ACL_INT64;
    case LLAISYS_DTYPE_U8: return ACL_UINT8;
    case LLAISYS_DTYPE_U16: return ACL_UINT16;
    case LLAISYS_DTYPE_U32: return ACL_UINT32;
    case LLAISYS_DTYPE_U64: return ACL_UINT64;
    case LLAISYS_DTYPE_F16: return ACL_FLOAT16;
    case LLAISYS_DTYPE_F32: return ACL_FLOAT;
    case LLAISYS_DTYPE_F64: return ACL_DOUBLE;
    case LLAISYS_DTYPE_BF16: return ACL_BF16;
    default: throw std::runtime_error("unsupported LLAISYS dtype for Ascend");
    }
}

std::vector<int64_t> contiguousStrides(const std::vector<int64_t> &shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) strides[i - 2] = strides[i - 1] * shape[i - 1];
    return strides;
}

AclTensorDesc::AclTensorDesc(void *data, llaisysDataType_t dtype, const std::vector<int64_t> &shape)
    : AclTensorDesc(data, dtype, shape, contiguousStrides(shape), shape) {}

AclTensorDesc::AclTensorDesc(void *data, llaisysDataType_t dtype, const std::vector<int64_t> &shape,
                             const std::vector<int64_t> &strides, const std::vector<int64_t> &storage_shape) {
    tensor_ = aclCreateTensor(shape.data(), shape.size(), toAclDataType(dtype), strides.data(), 0, ACL_FORMAT_ND,
                              storage_shape.data(), storage_shape.size(), data);
    if (!tensor_) throw std::runtime_error("aclCreateTensor returned null");
}

AclTensorDesc::~AclTensorDesc() {
    if (tensor_) (void)aclDestroyTensor(tensor_);
}

DeviceBuffer::DeviceBuffer(size_t bytes) {
    if (bytes) checkAcl(aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(temp)");
}

DeviceBuffer::~DeviceBuffer() {
    if (data_) (void)aclrtFree(data_);
}

aclrtStream currentStream() {
    return reinterpret_cast<aclrtStream>(llaisys::core::context().runtime().stream());
}

void runAclnn(uint64_t workspace_size, aclOpExecutor *executor, const char *name,
              const std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, aclrtStream)> &launch) {
    // Executors returned by an ACLNN GetWorkspaceSize call are owned by the
    // per-thread ACLNN allocator.  In CANN 9.0 they must not be passed to
    // aclDestroyAclOpExecutor: doing so corrupts that allocator and the next
    // GetWorkspaceSize call can crash in libnnopbase.
    void *workspace = workspace_size ? workspaceCache().acquire(workspace_size) : nullptr;
    checkAclnn(launch(workspace, workspace_size, executor, currentStream()), name);
    checkAcl(aclrtSynchronizeStream(currentStream()), "aclrtSynchronizeStream(op)");
}

} // namespace llaisys::device::ascend
