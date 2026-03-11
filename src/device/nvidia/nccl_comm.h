#pragma once

#ifdef ENABLE_NVIDIA_API

#include "llaisys.h"
#include <vector>
#include <memory>

namespace llaisys::device::nvidia {

class NcclComm {
public:
    NcclComm(const std::vector<int>& device_ids);
    ~NcclComm();

    NcclComm(const NcclComm&) = delete;
    NcclComm& operator=(const NcclComm&) = delete;

    void allreduceSum(const std::vector<void*>& bufs, size_t count, llaisysDataType_t dtype);
    void syncAll();

    int size() const;
    const std::vector<int>& deviceIds() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llaisys::device::nvidia

#endif // ENABLE_NVIDIA_API
