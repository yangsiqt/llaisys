#include "nvidia_resource.cuh"
#include <stdexcept>

namespace llaisys::device::nvidia {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    cublasCreate(&cublas_handle);
}

Resource::~Resource() {
    cublasDestroy(cublas_handle);
}

cublasHandle_t getCublasHandle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }
    }
    return handle;
}

} // namespace llaisys::device::nvidia
