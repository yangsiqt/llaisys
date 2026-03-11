#ifdef ENABLE_NVIDIA_API

#include "nccl_comm.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#define NCCL_CHECK(cmd) do {                                    \
    ncclResult_t r = cmd;                                       \
    if (r != ncclSuccess) {                                     \
        std::cerr << "NCCL error: " << ncclGetErrorString(r)    \
                  << " at " << __FILE__ << ":" << __LINE__      \
                  << std::endl;                                 \
        throw std::runtime_error("NCCL error");                 \
    }                                                           \
} while(0)

#define CUDA_CHECK(cmd) do {                                    \
    cudaError_t e = cmd;                                        \
    if (e != cudaSuccess) {                                     \
        std::cerr << "CUDA error: " << cudaGetErrorString(e)    \
                  << " at " << __FILE__ << ":" << __LINE__      \
                  << std::endl;                                 \
        throw std::runtime_error("CUDA error");                 \
    }                                                           \
} while(0)

namespace llaisys::device::nvidia {

struct NcclComm::Impl {
    int world_size;
    std::vector<int> device_ids;
    std::vector<ncclComm_t> comms;
    std::vector<cudaStream_t> streams;

    Impl(const std::vector<int>& dev_ids)
        : world_size(dev_ids.size()), device_ids(dev_ids) {

        comms.resize(world_size);
        streams.resize(world_size);

        NCCL_CHECK(ncclCommInitAll(comms.data(), world_size, device_ids.data()));

        for (int i = 0; i < world_size; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }

        std::cout << "[NCCL] Initialized " << world_size << " communicators for devices:";
        for (int id : device_ids) std::cout << " " << id;
        std::cout << std::endl;
    }

    ~Impl() {
        for (int i = 0; i < world_size; i++) {
            cudaSetDevice(device_ids[i]);
            cudaStreamDestroy(streams[i]);
            ncclCommDestroy(comms[i]);
        }
    }

    static ncclDataType_t toNcclType(llaisysDataType_t dtype) {
        switch (dtype) {
            case LLAISYS_DTYPE_F32: return ncclFloat;
            case LLAISYS_DTYPE_F16: return ncclHalf;
            case LLAISYS_DTYPE_BF16: return ncclBfloat16;
            case LLAISYS_DTYPE_F64: return ncclDouble;
            default:
                throw std::runtime_error("Unsupported dtype for NCCL allreduce");
        }
    }

    void allreduceSum(const std::vector<void*>& bufs, size_t count, llaisysDataType_t dtype) {
        ncclDataType_t nccl_type = toNcclType(dtype);

        for (int i = 0; i < world_size; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < world_size; i++) {
            NCCL_CHECK(ncclAllReduce(bufs[i], bufs[i], count, nccl_type,
                                     ncclSum, comms[i], streams[i]));
        }
        NCCL_CHECK(ncclGroupEnd());

        for (int i = 0; i < world_size; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
    }

    void syncAll() {
        for (int i = 0; i < world_size; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
};

NcclComm::NcclComm(const std::vector<int>& device_ids)
    : impl_(std::make_unique<Impl>(device_ids)) {}

NcclComm::~NcclComm() = default;

void NcclComm::allreduceSum(const std::vector<void*>& bufs, size_t count, llaisysDataType_t dtype) {
    impl_->allreduceSum(bufs, count, dtype);
}

void NcclComm::syncAll() {
    impl_->syncAll();
}

int NcclComm::size() const {
    return impl_->world_size;
}

const std::vector<int>& NcclComm::deviceIds() const {
    return impl_->device_ids;
}

} // namespace llaisys::device::nvidia

#endif // ENABLE_NVIDIA_API
