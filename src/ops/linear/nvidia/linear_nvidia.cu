// linear_nvidia.cu — BF16/F16：m、k 为 8 倍数时优先 CUTLASS TensorOp；先尝试真实 n（避免 decode 时 n=1 填成 8 的 ~8× 算力浪费），失败再零填充。否则 cuBLAS。FP32：cuBLAS TF32
#include "linear_nvidia.hpp"
#include "../../../device/nvidia/nvidia_resource.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/numeric_types.h>

__global__ void add_bias_f32(float *out, const float *bias, size_t n, size_t cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += bias[i % cols];
}
__global__ void add_bias_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *bias, size_t n, size_t cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(out[i]) + __bfloat162float(bias[i % cols]));
}
__global__ void add_bias_f16(__half *out, const __half *bias, size_t n, size_t cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(out[i]) + __half2float(bias[i % cols]));
}

namespace llaisys::ops::nvidia {
namespace {

void *d_gemm_scratch = nullptr;
size_t gemm_scratch_bytes = 0;

bool gemm_ensure_scratch(size_t bytes) {
    if (bytes <= gemm_scratch_bytes) return true;
    if (d_gemm_scratch) {
        cudaFree(d_gemm_scratch);
        d_gemm_scratch = nullptr;
        gemm_scratch_bytes = 0;
    }
    if (cudaMalloc(&d_gemm_scratch, bytes) != cudaSuccess) return false;
    gemm_scratch_bytes = bytes;
    return true;
}

/// 默认大 tile（高吞吐，适合较大 M）
using CutlassBf16TensorGemm =
    cutlass::gemm::device::Gemm<cutlass::bfloat16_t, cutlass::layout::RowMajor, cutlass::bfloat16_t,
                                cutlass::layout::ColumnMajor, cutlass::bfloat16_t,
                                cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
                                cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
                                cutlass::gemm::GemmShape<64, 64, 32>,
                                cutlass::gemm::GemmShape<16, 8, 16>,
                                cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8,
                                                                             float, float>,
                                cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 8,
                                8>;

using CutlassF16TensorGemm =
    cutlass::gemm::device::Gemm<cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
                                cutlass::layout::ColumnMajor, cutlass::half_t,
                                cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
                                cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
                                cutlass::gemm::GemmShape<64, 64, 32>,
                                cutlass::gemm::GemmShape<16, 8, 16>,
                                cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float,
                                                                             float>,
                                cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 8,
                                8>;

/// 较小 threadblock，部分小 M 问题在 can_implement 上更容易通过（仍 TensorOp）
using CutlassBf16TensorGemmSmallM =
    cutlass::gemm::device::Gemm<cutlass::bfloat16_t, cutlass::layout::RowMajor, cutlass::bfloat16_t,
                                cutlass::layout::ColumnMajor, cutlass::bfloat16_t,
                                cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
                                cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 64>,
                                cutlass::gemm::GemmShape<32, 32, 64>,
                                cutlass::gemm::GemmShape<16, 8, 16>,
                                cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8,
                                                                             float, float>,
                                cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 8,
                                8>;

using CutlassF16TensorGemmSmallM =
    cutlass::gemm::device::Gemm<cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
                                cutlass::layout::ColumnMajor, cutlass::half_t,
                                cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
                                cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 64>,
                                cutlass::gemm::GemmShape<32, 32, 64>,
                                cutlass::gemm::GemmShape<16, 8, 16>,
                                cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float,
                                                                             float>,
                                cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 8,
                                8>;

template <typename Gemm>
bool cutlass_gemm_run(Gemm &gemm, int n, int m, int k, const typename Gemm::ElementA *A,
                      const typename Gemm::ElementB *B, typename Gemm::ElementC *D) {
    float alpha = 1.f, beta = 0.f;
    typename Gemm::Arguments args({n, m, k}, {A, k}, {B, k}, {D, m}, {D, m}, {alpha, beta});
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    return gemm(args) == cutlass::Status::kSuccess;
}

/// m、k 须为 8 倍数。先直接 (n,m,k) TensorOp（n<16 先小 tile 再大 tile，否则相反），失败再 n 填充至 8 倍数仍走 CUTLASS
template <typename GemmLarge, typename GemmSmall, typename Element>
bool linear_cutlass_tensor(std::byte *out, const std::byte *in, const std::byte *weight, int n, int m,
                         int k) {
    if ((m & 7) != 0 || (k & 7) != 0) return false;
    if (n <= 0 || m <= 0 || k <= 0) return false;

    const Element *A_in = reinterpret_cast<const Element *>(in);
    const Element *B = reinterpret_cast<const Element *>(weight);
    Element *D_out = reinterpret_cast<Element *>(out);
    float alpha = 1.f, beta = 0.f;

    static GemmLarge gemm_large;
    static GemmSmall gemm_small;
    // decode 常见 n=1：先小 tile，少一次大 tile 的 can_implement；长 prefill 先大 tile
    if (n < 16) {
        if (cutlass_gemm_run(gemm_small, n, m, k, A_in, B, D_out)) return true;
        if (cutlass_gemm_run(gemm_large, n, m, k, A_in, B, D_out)) return true;
    } else {
        if (cutlass_gemm_run(gemm_large, n, m, k, A_in, B, D_out)) return true;
        if (cutlass_gemm_run(gemm_small, n, m, k, A_in, B, D_out)) return true;
    }

    const int n_pad = (n + 7) & ~7;
    if (n_pad == n) return false;

    const size_t row_bytes = (size_t)k * sizeof(Element);
    const size_t need_a = (size_t)n_pad * k * sizeof(Element);
    const size_t need_d = (size_t)n_pad * m * sizeof(Element);
    if (!gemm_ensure_scratch(need_a + need_d)) return false;

    auto *scrA = reinterpret_cast<Element *>(d_gemm_scratch);
    auto *scrD = reinterpret_cast<Element *>(reinterpret_cast<std::byte *>(d_gemm_scratch) + need_a);

    if (cudaMemcpyAsync(scrA, A_in, (size_t)n * row_bytes, cudaMemcpyDeviceToDevice) !=
        cudaSuccess)
        return false;
    if (cudaMemsetAsync(reinterpret_cast<std::byte *>(scrA) + (size_t)n * row_bytes, 0,
                        (size_t)(n_pad - n) * row_bytes) != cudaSuccess)
        return false;

    typename GemmLarge::Arguments args({n_pad, m, k}, {scrA, k}, {B, k}, {scrD, m}, {scrD, m},
                                       {alpha, beta});
    if (gemm_large.can_implement(args) != cutlass::Status::kSuccess) return false;
    if (gemm_large(args) != cutlass::Status::kSuccess) return false;

    if (cudaMemcpyAsync(D_out, scrD, (size_t)n * (size_t)m * sizeof(Element),
                        cudaMemcpyDeviceToDevice) != cudaSuccess)
        return false;
    return true;
}

} // namespace

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features) {
    cublasHandle_t handle = llaisys::device::nvidia::getCublasHandle();
    int m = (int)out_features, n = (int)batch_size, k = (int)in_features;
    int thr = 256;
    size_t elems = batch_size * out_features;

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        float alpha = 1.f, beta = 0.f;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, weight, CUDA_R_32F, k, in,
                     CUDA_R_32F, k, &beta, out, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_TF32,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (bias)
            add_bias_f32<<<((int)elems + thr - 1) / thr, thr>>>((float *)out, (const float *)bias,
                                                                 elems, out_features);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        bool ok = linear_cutlass_tensor<CutlassBf16TensorGemm, CutlassBf16TensorGemmSmallM,
                                        cutlass::bfloat16_t>(out, in, weight, n, m, k);
        if (!ok) {
            float alpha = 1.f, beta = 0.f;
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, weight, CUDA_R_16BF, k,
                         in, CUDA_R_16BF, k, &beta, out, CUDA_R_16BF, m, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        if (bias)
            add_bias_bf16<<<((int)elems + thr - 1) / thr, thr>>>(
                (__nv_bfloat16 *)out, (const __nv_bfloat16 *)bias, elems, out_features);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        bool ok = linear_cutlass_tensor<CutlassF16TensorGemm, CutlassF16TensorGemmSmallM,
                                        cutlass::half_t>(out, in, weight, n, m, k);
        if (!ok) {
            float alpha = 1.f, beta = 0.f;
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, weight, CUDA_R_16F, k,
                         in, CUDA_R_16F, k, &beta, out, CUDA_R_16F, m, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        if (bias)
            add_bias_f16<<<((int)elems + thr - 1) / thr, thr>>>(
                (__half *)out, (const __half *)bias, elems, out_features);
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for CUDA linear");
    }
}
} // namespace llaisys::ops::nvidia
