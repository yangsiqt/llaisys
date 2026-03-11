// ============================================================================
//  gemm_cutlass.cu  ——  使用 CUTLASS 实现 BF16 矩阵乘法
//
//  计算:  D[M,N] = α × A[M,K] × B[K,N] + β × C[M,N]
//         (BF16 输入, FP32 累加, BF16 输出, Tensor Core 加速)
//
//  与 gemm_cuda.cu 做对比, 展示 CUTLASS 的核心概念:
//
//  ┌─────────────────────────────────────────────────────────────────┐
//  │  CUTLASS 三级分块架构                                           │
//  │                                                                 │
//  │  ThreadblockShape  ─── 一个 CTA (block) 负责的输出子矩阵       │
//  │       ↓                                                         │
//  │  WarpShape         ─── 一个 warp 负责的子矩阵                  │
//  │       ↓                                                         │
//  │  InstructionShape  ─── 一条 MMA 指令处理的子矩阵 (Tensor Core) │
//  └─────────────────────────────────────────────────────────────────┘
//
//  关键头文件路径 (相对 cutlass/include/):
//    cutlass/gemm/device/gemm.h          设备级 GEMM 接口
//    cutlass/epilogue/thread/linear_combination.h   epilogue: D = α*acc + β*C
//    cutlass/numeric_types.h             bfloat16_t / half_t
//    cutlass/layout/matrix.h             RowMajor / ColumnMajor
//
//  编译 (从 test_cutlass/ 目录):
//    nvcc -O3 -arch=sm_86 -std=c++17 \
//         -I../third_party/cutlass/include \
//         --expt-relaxed-constexpr \
//         gemm_cutlass.cu -o gemm_cutlass
//  运行:
//    ./gemm_cutlass
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ===== CUTLASS 头文件 =====
#include <cutlass/cutlass.h>                           // Status, 基本类型
#include <cutlass/numeric_types.h>                     // bfloat16_t
#include <cutlass/gemm/device/gemm.h>                  // device::Gemm 模板
#include <cutlass/epilogue/thread/linear_combination.h>// epilogue 策略
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

// ====================== 工具宏 ======================
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d : %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

void fill_random_bf16(__nv_bfloat16 *dst, size_t n) {
    for (size_t i = 0; i < n; i++)
        dst[i] = __float2bfloat16((float)(rand() % 100) / 100.f - 0.5f);
}

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

float max_abs_error(const __nv_bfloat16 *gpu_result, const float *ref,
                    int M, int N) {
    float mx = 0.f;
    for (int i = 0; i < M * N; i++) {
        float d = fabsf(__bfloat162float(gpu_result[i]) - ref[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// ============================================================================
//  第一步: 定义 CUTLASS GEMM 类型
//
//  cutlass::gemm::device::Gemm<...> 是一个 "配置即代码" 的模板:
//  你通过模板参数选好所有策略, CUTLASS 在编译期生成对应的高性能 kernel.
//
//  模板参数含义 (按顺序):
//
//    ElementA, LayoutA       —— A 矩阵的元素类型和存储布局
//    ElementB, LayoutB       —— B 矩阵
//    ElementC, LayoutC       —— C/D 矩阵 (输出)
//    ElementAccumulator      —— 累加器类型 (通常 float, 精度更高)
//    OperatorClass           —— 使用什么计算单元:
//                                 OpClassSimt     → CUDA Core (FP32)
//                                 OpClassTensorOp → Tensor Core (快!)
//    ArchTag                 —— 目标架构: Sm80 = Ampere (覆盖 sm_86)
//    ThreadblockShape        —— CTA 级别分块 {M, N, K}
//    WarpShape               —— Warp 级别分块 {M, N, K}
//    InstructionShape        —— 单条 MMA 指令 {M, N, K}
//                                 BF16 Tensor Core = {16, 8, 16}
//    EpilogueOp              —— 输出处理: D = α × accumulator + β × C
//    ThreadblockSwizzle      —— CTA 到 output tile 的映射策略
//    Stages                  —— 流水线深度 (越大隐藏越多访存延迟)
//    AlignmentA, AlignmentB  —— 向量化宽度 (8 = 每次读 8 个 BF16 = 16 bytes)
// ============================================================================

using CutlassBf16Gemm = cutlass::gemm::device::Gemm<
    // ---- 矩阵元素类型和布局 ----
    cutlass::bfloat16_t,                       // ElementA
    cutlass::layout::RowMajor,                 // LayoutA   : A[M,K] 按行存储
    cutlass::bfloat16_t,                       // ElementB
    cutlass::layout::RowMajor,                 // LayoutB   : B[K,N] 按行存储
    cutlass::bfloat16_t,                       // ElementC/D
    cutlass::layout::RowMajor,                 // LayoutC/D : C[M,N] 按行存储

    // ---- 累加器和计算单元 ----
    float,                                     // ElementAccumulator = FP32
    cutlass::arch::OpClassTensorOp,            // 使用 Tensor Core
    cutlass::arch::Sm80,                       // Ampere 架构 (sm_80/86)

    // ---- 三级分块尺寸 ----
    //
    //  ThreadblockShape = {128, 128, 32}
    //    → 每个 CTA 输出 128×128 的子矩阵
    //    → 每次从 K 维度取 32 列做乘加
    //
    //  WarpShape = {64, 64, 32}
    //    → 每个 warp 负责 64×64 的子矩阵
    //    → CTA 内有 (128/64)×(128/64) = 4 个 warp
    //
    //  InstructionShape = {16, 8, 16}
    //    → BF16 Tensor Core MMA 指令的固有尺寸
    //    → 一条指令: 16×8 输出, 消耗 K=16 个元素
    //
    cutlass::gemm::GemmShape<128, 128, 32>,    // ThreadblockShape
    cutlass::gemm::GemmShape<64,  64,  32>,    // WarpShape
    cutlass::gemm::GemmShape<16,   8,  16>,    // InstructionShape

    // ---- Epilogue (后处理) ----
    //  D = α × accumulator + β × C
    //  参数: <输出类型, 向量宽度, 累加器类型, 计算类型>
    //  向量宽度 = 128 bits / 16 bits = 8 个 BF16 元素
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t,                   // 输出元素类型
        8,                                     // 向量宽度
        float,                                 // 累加器类型
        float                                  // 计算类型
    >,

    // ---- 线程块映射策略 ----
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,

    // ---- 流水线级数 ----
    3,                                         // Stages (3-stage pipeline)

    // ---- 对齐 ----
    8,                                         // AlignmentA (8 × BF16 = 16 bytes)
    8                                          // AlignmentB
>;

// ============================================================================
//  第二步: 运行 CUTLASS GEMM
//
//  CUTLASS 的使用流程:
//    1. 构造 Arguments (问题描述)
//    2. 创建 Gemm 对象
//    3. 调用 can_implement() 检查硬件兼容性
//    4. 调用 operator()  启动 kernel
// ============================================================================
void run_cutlass(const __nv_bfloat16 *dA, const __nv_bfloat16 *dB,
                 __nv_bfloat16 *dD, int M, int N, int K) {

    // 类型别名 (CUTLASS 内部的 bfloat16_t 和 CUDA 的 __nv_bfloat16 内存布局一致)
    using Element = cutlass::bfloat16_t;
    const auto *A = reinterpret_cast<const Element *>(dA);
    const auto *B = reinterpret_cast<const Element *>(dB);
    auto       *D = reinterpret_cast<Element *>(dD);

    float alpha = 1.0f;
    float beta  = 0.0f;   // 纯 GEMM, 不加 C

    // ---- 1. 构造参数 ----
    //
    //  Arguments(problem_size, ref_A, ref_B, ref_C, ref_D, epilogue_params)
    //
    //  ref_X = {指针, leading_dimension}
    //    - RowMajor [M,K] 的 leading dimension = K (一行有 K 个元素)
    //    - RowMajor [K,N] 的 leading dimension = N
    //
    CutlassBf16Gemm::Arguments args(
        {M, N, K},                  // GemmCoord problem_size
        {A, K},                     // TensorRef A [M,K], lda = K
        {B, N},                     // TensorRef B [K,N], ldb = N
        {D, N},                     // TensorRef C (不用, beta=0)
        {D, N},                     // TensorRef D [M,N], ldd = N
        {alpha, beta}               // epilogue: D = 1.0*acc + 0.0*C
    );

    // ---- 2. 实例化 Gemm 对象 ----
    CutlassBf16Gemm gemm_op;

    // ---- 3. 检查是否能执行 ----
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS: can_implement 失败 (%s)\n",
                cutlass::cutlassGetStatusString(status));
        exit(EXIT_FAILURE);
    }

    // ---- 4. 启动 kernel ----
    status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS: 执行失败 (%s)\n",
                cutlass::cutlassGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
//  计时
// ============================================================================
float benchmark_cutlass(const __nv_bfloat16 *dA, const __nv_bfloat16 *dB,
                        __nv_bfloat16 *dD, int M, int N, int K,
                        int warmup, int repeat) {
    for (int i = 0; i < warmup; i++) run_cutlass(dA, dB, dD, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));
    for (int i = 0; i < repeat; i++) run_cutlass(dA, dB, dD, M, N, K);
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    return ms / repeat;
}

// ============================================================================
int main() {
    constexpr int M = 32;
    constexpr int N = 8960;
    constexpr int K = 1536;

    printf("=== CUTLASS BF16 GEMM 示例 (Tensor Core) ===\n");
    printf("问题规模: M=%d, N=%d, K=%d  (BF16, FP32 累加)\n", M, N, K);
    printf("Threadblock={128,128,32}  Warp={64,64,32}  MMA={16,8,16}\n\n");

    // ---- host 数据 ----
    size_t sA = M * K, sB = K * N, sC = M * N;
    auto *hA  = new __nv_bfloat16[sA];
    auto *hB  = new __nv_bfloat16[sB];
    auto *hC  = new __nv_bfloat16[sC];
    auto *ref = new float[sC];

    srand(42);
    fill_random_bf16(hA, sA);
    fill_random_bf16(hB, sB);

    printf("计算 CPU 参考结果...\n");
    gemm_ref_f32(hA, hB, ref, M, N, K);

    // ---- device 数据 ----
    __nv_bfloat16 *dA, *dB, *dD;
    CHECK_CUDA(cudaMalloc(&dA, sA * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dB, sB * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dD, sC * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemcpy(dA, hA, sA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, sB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // ---- 运行 CUTLASS GEMM ----
    float ms = benchmark_cutlass(dA, dB, dD, M, N, K, 5, 20);
    CHECK_CUDA(cudaMemcpy(hC, dD, sC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    float err = max_abs_error(hC, ref, M, N);
    double gflops = 2.0 * M * N * K / (ms * 1e6);

    printf("\n[CUTLASS GEMM]  %.3f ms  |  %.1f GFLOPS  |  max_err=%.4f\n", ms, gflops, err);

    // ---- 对比: 3090 理论峰值 ----
    //   BF16 Tensor Core: 142 TFLOPS (结构化稀疏 284T, 这里不用)
    //   达到率 = 实测 GFLOPS / 142000
    double util = gflops / 142000.0 * 100.0;
    printf("3090 BF16 Tensor Core 利用率: %.1f%%\n", util);

    printf("\n比较: 回去运行 gemm_cuda 看 Naive/Tiled 的 GFLOPS,\n"
           "      就能感受 Tensor Core 的碾压级差距!\n");

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dD));
    delete[] hA; delete[] hB; delete[] hC; delete[] ref;
    return 0;
}
