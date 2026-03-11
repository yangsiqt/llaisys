#include "nvidia_resource.cuh"
#include <stdexcept>
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>

namespace llaisys::device::nvidia {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    cudaSetDevice(device_id);
    cublasCreate(&cublas_handle);
}

Resource::~Resource() {
    cublasDestroy(cublas_handle);
}

cublasHandle_t getCublasHandle() {
    static std::unordered_map<int, cublasHandle_t> handles;
    static std::mutex mu;

    int device_id = 0;
    cudaGetDevice(&device_id);

    std::lock_guard<std::mutex> lock(mu);
    auto it = handles.find(device_id);
    if (it != handles.end()) return it->second;

    cublasHandle_t handle = nullptr;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS handle");
    }
    handles[device_id] = handle;
    return handle;
}

} // namespace llaisys::device::nvidia
