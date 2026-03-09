#include "argmax_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <stdexcept>

// Single-block grid-stride argmax with shared memory reduction
template<typename T>
__device__ inline float to_float(T v);
template<> __device__ inline float to_float<float>(float v) { return v; }
template<> __device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template<> __device__ inline float to_float<__half>(__half v) { return __half2float(v); }

template<typename TVal, typename TOut>
__device__ inline void write_max(TOut *ptr, float v);
template<> __device__ inline void write_max<float,float>(float *p, float v) { *p = v; }
template<> __device__ inline void write_max<__nv_bfloat16,__nv_bfloat16>(__nv_bfloat16 *p, float v) { *p = __float2bfloat16(v); }
template<> __device__ inline void write_max<__half,__half>(__half *p, float v) { *p = __float2half(v); }

template<typename T>
__global__ void argmax_kernel(int64_t *out_idx, T *out_val, const T *vals, size_t n) {
    extern __shared__ char smem[];
    float   *s_vals = reinterpret_cast<float*>(smem);
    int64_t *s_idxs = reinterpret_cast<int64_t*>(s_vals + blockDim.x);

    size_t tid = threadIdx.x;
    float local_max = -FLT_MAX;
    int64_t local_idx = 0;

    for (size_t i = tid; i < n; i += blockDim.x) {
        float v = to_float(vals[i]);
        if (v > local_max) { local_max = v; local_idx = (int64_t)i; }
    }
    s_vals[tid] = local_max; s_idxs[tid] = local_idx;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s && s_vals[tid+s] > s_vals[tid]) {
            s_vals[tid] = s_vals[tid+s]; s_idxs[tid] = s_idxs[tid+s];
        }
        __syncthreads();
    }
    if (tid == 0) { out_idx[0] = s_idxs[0]; write_max<T,T>(out_val, s_vals[0]); }
}

namespace llaisys::ops::nvidia {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t numel) {
    int threads = 1024;
    size_t smem = threads * (sizeof(float) + sizeof(int64_t));
    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<float><<<1,threads,smem>>>(reinterpret_cast<int64_t*>(max_idx),reinterpret_cast<float*>(max_val),reinterpret_cast<const float*>(vals),numel); break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<__nv_bfloat16><<<1,threads,smem>>>(reinterpret_cast<int64_t*>(max_idx),reinterpret_cast<__nv_bfloat16*>(max_val),reinterpret_cast<const __nv_bfloat16*>(vals),numel); break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<__half><<<1,threads,smem>>>(reinterpret_cast<int64_t*>(max_idx),reinterpret_cast<__half*>(max_val),reinterpret_cast<const __half*>(vals),numel); break;
    default: throw std::runtime_error("Unsupported dtype for CUDA argmax");
    }
}
} // namespace llaisys::ops::nvidia
