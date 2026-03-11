// ============================================================================
//  gemm_cuda.cu  ——  原生 CUDA 实现 BF16 矩阵乘法
//
//  计算:  C[M,N] = A[M,K] × B[K,N]       (BF16 输入, FP32 累加, BF16 输出)
//
//  包含两个版本:
//    1. Naive kernel : 每个线程计算 C 的一个元素 (教学用, 性能差)
//    2. Tiled kernel : Shared Memory 分块 + 向量化 (接近实用水平)
//
//  编译:
//    nvcc -O3 -arch=sm_86 -std=c++17 gemm_cuda.cu -o gemm_cuda
//  运行:
//    ./gemm_cuda
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// ====================== 工具函数 ======================

// 检查 CUDA 调用是否成功
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d : %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// 在 host 端生成随机 BF16 数据 (先生成 float 再转换)
void fill_random_bf16(__nv_bfloat16 *dst, size_t n) {
    for (size_t i = 0; i < n; i++)
        dst[i] = __float2bfloat16((float)(rand() % 100) / 100.f - 0.5f);
}

// host 端 FP32 参考 GEMM (用于验证正确性)
void gemm_ref_f32(const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C,
                  int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += __bfloat162float(A[m * K + k]) *
                       __bfloat162float(B[k * N + n]);
            C[m * N + n] = acc;
        }
}

// ====================== Kernel 1: Naive GEMM ======================
//
// 最简单的 GEMM: 每个线程负责 C 中的一个元素
// 线程 (row, col) 计算 C[row][col] = Σ_k A[row][k] * B[k][col]
//
// 性能瓶颈:
//   - 每个线程独立读取一整行 A 和一整列 B → 大量重复的 global memory 读取
//   - 没有利用数据局部性
//
__global__ void gemm_naive_bf16(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                                __nv_bfloat16 *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; k++) {
        acc += __bfloat162float(A[row * K + k]) *
               __bfloat162float(B[k * N + col]);
    }
    C[row * N + col] = __float2bfloat16(acc);
}

// ====================== Kernel 2: Tiled GEMM ======================
//
// Shared Memory 分块优化:
//   - 将 A、B 分成 TILE_SIZE × TILE_SIZE 的小块
//   - 每个 block 协作地把一个 tile 加载到 shared memory
//   - 然后所有线程从 shared memory (低延迟) 计算部分和
//   - 循环所有 K 方向的 tile
//
// 优势:
//   - 每个元素从 global memory 只读一次, 在 shared memory 中被复用 TILE_SIZE 次
//   - global memory 访问量降低 TILE_SIZE 倍
//
static constexpr int TILE_SIZE = 32;

__global__ void gemm_tiled_bf16(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                                __nv_bfloat16 *C, int M, int N, int K) {
    // Shared memory: 每个 block 持有 A 和 B 各一个 tile
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.f;

    // 沿 K 维度滑动 tile
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        // ---- 协作加载 tile 到 shared memory ----
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? __bfloat162float(A[row * K + a_col]) : 0.f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? __bfloat162float(B[b_row * N + col]) : 0.f;
        __syncthreads();

        // ---- 从 shared memory 计算部分乘积 ----
        for (int i = 0; i < TILE_SIZE; i++)
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = __float2bfloat16(acc);
}

// ====================== 计时辅助 ======================

float benchmark_kernel(void (*launcher)(const __nv_bfloat16 *, const __nv_bfloat16 *,
                                        __nv_bfloat16 *, int, int, int),
                       const __nv_bfloat16 *dA, const __nv_bfloat16 *dB,
                       __nv_bfloat16 *dC, int M, int N, int K,
                       int warmup, int repeat) {
    for (int i = 0; i < warmup; i++) launcher(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));
    for (int i = 0; i < repeat; i++) launcher(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    return ms / repeat;
}

// launcher 包装
void launch_naive(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                  __nv_bfloat16 *C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_naive_bf16<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_tiled(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                  __nv_bfloat16 *C, int M, int N, int K) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_tiled_bf16<<<grid, block>>>(A, B, C, M, N, K);
}

// ====================== 验证 ======================

float max_abs_error(const __nv_bfloat16 *gpu_result, const float *ref,
                    int M, int N) {
    float max_err = 0.f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(__bfloat162float(gpu_result[i]) - ref[i]);
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

// ====================== main ======================

int main() {
    // ---- 问题规模 (和 Qwen2-1.5B 的 gate/up projection 相同) ----
    constexpr int M = 32;     // batch size (prefill token 数)
    constexpr int N = 8960;   // out_features (intermediate_size)
    constexpr int K = 1536;   // in_features  (hidden_size)

    printf("=== 原生 CUDA BF16 GEMM 示例 ===\n");
    printf("问题规模: M=%d, N=%d, K=%d  (BF16)\n\n", M, N, K);

    // ---- 分配 host 内存 ----
    size_t sA = M * K, sB = K * N, sC = M * N;
    auto *hA = new __nv_bfloat16[sA];
    auto *hB = new __nv_bfloat16[sB];
    auto *hC = new __nv_bfloat16[sC];
    auto *ref = new float[sC];

    srand(42);
    fill_random_bf16(hA, sA);
    fill_random_bf16(hB, sB);

    // ---- CPU 参考结果 ----
    printf("计算 CPU 参考结果...\n");
    gemm_ref_f32(hA, hB, ref, M, N, K);

    // ---- 分配 device 内存 ----
    __nv_bfloat16 *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, sA * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dB, sB * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC, sC * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemcpy(dA, hA, sA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, sB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // ---- Naive GEMM ----
    float ms_naive = benchmark_kernel(launch_naive, dA, dB, dC, M, N, K, 3, 10);
    CHECK_CUDA(cudaMemcpy(hC, dC, sC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    float err_naive = max_abs_error(hC, ref, M, N);
    double gflops_naive = 2.0 * M * N * K / (ms_naive * 1e6);
    printf("[Naive  GEMM]  %.3f ms  |  %.1f GFLOPS  |  max_err=%.4f\n",
           ms_naive, gflops_naive, err_naive);

    // ---- Tiled GEMM ----
    float ms_tiled = benchmark_kernel(launch_tiled, dA, dB, dC, M, N, K, 3, 10);
    CHECK_CUDA(cudaMemcpy(hC, dC, sC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    float err_tiled = max_abs_error(hC, ref, M, N);
    double gflops_tiled = 2.0 * M * N * K / (ms_tiled * 1e6);
    printf("[Tiled  GEMM]  %.3f ms  |  %.1f GFLOPS  |  max_err=%.4f\n",
           ms_tiled, gflops_tiled, err_tiled);

    printf("\nTiled vs Naive 加速比: %.1fx\n", ms_naive / ms_tiled);

    // ---- 清理 ----
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    delete[] hA; delete[] hB; delete[] hC; delete[] ref;
    return 0;
}
