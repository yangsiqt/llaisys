#pragma once

#include "llaisys.h"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace llaisys::device::ascend {

void checkAcl(aclError status, const char *call);
void checkAclnn(aclnnStatus status, const char *call);
aclDataType toAclDataType(llaisysDataType_t dtype);
std::vector<int64_t> contiguousStrides(const std::vector<int64_t> &shape);

class AclTensorDesc {
public:
    AclTensorDesc(void *data, llaisysDataType_t dtype, const std::vector<int64_t> &shape);
    AclTensorDesc(void *data, llaisysDataType_t dtype, const std::vector<int64_t> &shape,
                  const std::vector<int64_t> &strides, const std::vector<int64_t> &storage_shape);
    ~AclTensorDesc();
    AclTensorDesc(const AclTensorDesc &) = delete;
    AclTensorDesc &operator=(const AclTensorDesc &) = delete;
    aclTensor *get() const { return tensor_; }

private:
    aclTensor *tensor_ = nullptr;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    void *data() const { return data_; }

private:
    void *data_ = nullptr;
};

aclrtStream currentStream();
void runAclnn(uint64_t workspace_size, aclOpExecutor *executor, const char *name,
              const std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, aclrtStream)> &launch);

} // namespace llaisys::device::ascend
