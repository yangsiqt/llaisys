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

template<typename T>
__global__ void argmax_batch_kernel(int64_t *out_idx, T *out_val, const T *vals, size_t width) {
    extern __shared__ char smem[];
    float   *s_vals = reinterpret_cast<float*>(smem);
    int64_t *s_idxs = reinterpret_cast<int64_t*>(s_vals + blockDim.x);

    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    const T *row_vals = vals + row * width;
    float local_max = -FLT_MAX;
    int64_t local_idx = 0;

    for (size_t i = tid; i < width; i += blockDim.x) {
        float v = to_float(row_vals[i]);
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
    if (tid == 0) {
        out_idx[row] = s_idxs[0];
        write_max<T,T>(out_val + row, s_vals[0]);
    }
}

template<typename T>
__global__ void argmax_batch_partial_kernel(int64_t *partial_idx, T *partial_val,
                                            const T *vals, size_t width, size_t num_parts) {
    extern __shared__ char smem[];
    float   *s_vals = reinterpret_cast<float*>(smem);
    int64_t *s_idxs = reinterpret_cast<int64_t*>(s_vals + blockDim.x);

    size_t row = blockIdx.x;
    size_t part = blockIdx.y;
    size_t tid = threadIdx.x;
    size_t part_size = (width + num_parts - 1) / num_parts;
    size_t begin = part * part_size;
    size_t end = begin + part_size < width ? begin + part_size : width;
    const T *row_vals = vals + row * width;

    float local_max = -FLT_MAX;
    int64_t local_idx = 0;
    for (size_t i = begin + tid; i < end; i += blockDim.x) {
        float v = to_float(row_vals[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = (int64_t)i;
        }
    }

    s_vals[tid] = local_max;
    s_idxs[tid] = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_vals[tid + s] > s_vals[tid]) {
            s_vals[tid] = s_vals[tid + s];
            s_idxs[tid] = s_idxs[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        size_t out = row * num_parts + part;
        partial_idx[out] = s_idxs[0];
        write_max<T,T>(partial_val + out, s_vals[0]);
    }
}

template<typename T>
__global__ void argmax_batch_finalize_kernel(int64_t *out_idx, T *out_val,
                                             const int64_t *partial_idx,
                                             const T *partial_val,
                                             size_t num_parts) {
    extern __shared__ char smem[];
    float   *s_vals = reinterpret_cast<float*>(smem);
    int64_t *s_idxs = reinterpret_cast<int64_t*>(s_vals + blockDim.x);

    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    const int64_t *idx_row = partial_idx + row * num_parts;
    const T *val_row = partial_val + row * num_parts;

    float local_max = -FLT_MAX;
    int64_t local_idx = 0;
    for (size_t i = tid; i < num_parts; i += blockDim.x) {
        float v = to_float(val_row[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = idx_row[i];
        }
    }

    s_vals[tid] = local_max;
    s_idxs[tid] = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_vals[tid + s] > s_vals[tid]) {
            s_vals[tid] = s_vals[tid + s];
            s_idxs[tid] = s_idxs[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_idx[row] = s_idxs[0];
        write_max<T,T>(out_val + row, s_vals[0]);
    }
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

void argmax_batch(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
                  llaisysDataType_t type, size_t batch_size, size_t width) {
    int threads = 1024;
    size_t smem = threads * (sizeof(float) + sizeof(int64_t));
    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_batch_kernel<float><<<(int)batch_size,threads,smem>>>(reinterpret_cast<int64_t*>(max_idx),reinterpret_cast<float*>(max_val),reinterpret_cast<const float*>(vals),width); break;
    case LLAISYS_DTYPE_BF16:
        argmax_batch_kernel<__nv_bfloat16><<<(int)batch_size,threads,smem>>>(reinterpret_cast<int64_t*>(max_idx),reinterpret_cast<__nv_bfloat16*>(max_val),reinterpret_cast<const __nv_bfloat16*>(vals),width); break;
    case LLAISYS_DTYPE_F16:
        argmax_batch_kernel<__half><<<(int)batch_size,threads,smem>>>(reinterpret_cast<int64_t*>(max_idx),reinterpret_cast<__half*>(max_val),reinterpret_cast<const __half*>(vals),width); break;
    default: throw std::runtime_error("Unsupported dtype for CUDA argmax_batch");
    }
}

void argmax_batch_fast(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
                       std::byte *partial_idx, std::byte *partial_val,
                       llaisysDataType_t type, size_t batch_size, size_t width,
                       size_t num_parts) {
    int partial_threads = 256;
    int final_threads = 256;
    size_t partial_smem = partial_threads * (sizeof(float) + sizeof(int64_t));
    size_t final_smem = final_threads * (sizeof(float) + sizeof(int64_t));
    dim3 partial_grid((unsigned)batch_size, (unsigned)num_parts);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_batch_partial_kernel<float><<<partial_grid, partial_threads, partial_smem>>>(
            reinterpret_cast<int64_t*>(partial_idx), reinterpret_cast<float*>(partial_val),
            reinterpret_cast<const float*>(vals), width, num_parts);
        argmax_batch_finalize_kernel<float><<<(int)batch_size, final_threads, final_smem>>>(
            reinterpret_cast<int64_t*>(max_idx), reinterpret_cast<float*>(max_val),
            reinterpret_cast<const int64_t*>(partial_idx), reinterpret_cast<const float*>(partial_val),
            num_parts);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_batch_partial_kernel<__nv_bfloat16><<<partial_grid, partial_threads, partial_smem>>>(
            reinterpret_cast<int64_t*>(partial_idx), reinterpret_cast<__nv_bfloat16*>(partial_val),
            reinterpret_cast<const __nv_bfloat16*>(vals), width, num_parts);
        argmax_batch_finalize_kernel<__nv_bfloat16><<<(int)batch_size, final_threads, final_smem>>>(
            reinterpret_cast<int64_t*>(max_idx), reinterpret_cast<__nv_bfloat16*>(max_val),
            reinterpret_cast<const int64_t*>(partial_idx), reinterpret_cast<const __nv_bfloat16*>(partial_val),
            num_parts);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_batch_partial_kernel<__half><<<partial_grid, partial_threads, partial_smem>>>(
            reinterpret_cast<int64_t*>(partial_idx), reinterpret_cast<__half*>(partial_val),
            reinterpret_cast<const __half*>(vals), width, num_parts);
        argmax_batch_finalize_kernel<__half><<<(int)batch_size, final_threads, final_smem>>>(
            reinterpret_cast<int64_t*>(max_idx), reinterpret_cast<__half*>(max_val),
            reinterpret_cast<const int64_t*>(partial_idx), reinterpret_cast<const __half*>(partial_val),
            num_parts);
        break;
    default: throw std::runtime_error("Unsupported dtype for CUDA argmax_batch_fast");
    }
}
} // namespace llaisys::ops::nvidia
