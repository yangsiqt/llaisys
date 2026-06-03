# LLAISYS：面向大模型 Serving 的 C++/CUDA 推理系统

LLAISYS 是一个面向大模型推理系统的 C++/CUDA 项目，覆盖 Qwen2 端到端推理、CUDA 算子、TP 张量并行、Continuous Batching、Paged KV Cache 与 online serving benchmark。

项目采用 Python 负责模型加载、权重切分与请求调度，通过 C API 调用 C++/CUDA 后端执行推理。整体链路从张量、算子、KV cache、batch scheduler 到 TP 通信逐层下沉，重点验证 GPU serving 场景中的吞吐、延迟、显存与通信开销。

---

## 核心特性

- **C++/CUDA 推理后端**：实现 Qwen2 系列模型的端到端推理链路，支持 FP32 / FP16 / BF16。
- **Python 前端 + C API**：使用 Python 对接 HuggingFace / Safetensors 生态，通过 ctypes 调用 C++ 后端。
- **CUDA 算子路径**：实现 Embedding、Linear、RMSNorm、RoPE、Self-Attention、SwiGLU 等算子。
- **cuBLAS / CUTLASS Linear**：针对 GPU Linear 路径适配 cuBLAS，并在 BF16 / FP16 对齐场景下支持 CUTLASS 路径。
- **TP=2 张量并行**：采用 Megatron-style QKV/O-Proj 与 SwiGLU-MLP 列/行并行，每层通过 NCCL AllReduce 聚合。
- **Continuous Batching**：为每个请求维护独立 slot，scheduler 每轮动态组成 active batch 进行 decode。
- **Paged KV Cache / Block Manager**：按 token block 动态分配 KV cache，支持 block table、block 释放复用与 utilization 统计。
- **Serving 优化路径**：实现 prefill workspace、paged decode attention、chunked prefill + decode interleave、CUDA Graph fallback、NCCL async allreduce 等实验与 A/B 路径。
- **Online Serving Benchmark**：统计 output tok/s、TTFT、TPOT、p99 latency、峰值显存、active batch、waiting queue 与 KV block 使用情况。

---

## 性能概览

测试环境：


| 项目  | 配置                                         |
| --- | ------------------------------------------ |
| 模型  | DeepSeek-R1-Distill-Qwen-14B               |
| 精度  | BF16                                       |
| GPU | 2 x A800 80GB                              |
| 场景  | p128d128 controlled serving                |
| 输入  | 真实 ShareGPT prompt token-id，截断到 128 tokens |
| 输出  | 固定生成 128 tokens，`ignore_eos=true`          |
| 请求  | 32 requests，arrival=inf，max concurrency=32 |


LLAISYS p128d128 controlled serving 最优性能：


| 系统      | 配置                              | Output tok/s | Total tok/s | 平均 TTFT | 平均 TPOT | P99 latency | 峰值显存      |
| ------- | ------------------------------- | ------------ | ----------- | ------- | ------- | ----------- | --------- |
| LLAISYS | TP=2 Paged KV + Paged Attention | 653          | 1306        | 1166 ms | 39.3 ms | 6154 ms     | 23349 MiB |


---

## 技术架构

```text
Python frontend
  ├── HuggingFace tokenizer / config
  ├── Safetensors weight loading
  ├── TP rank weight slicing
  ├── Continuous batching scheduler
  └── ctypes C API

C API
  ├── Qwen2 / Qwen2TP model lifecycle
  ├── init continuous / paged continuous
  ├── prefill / decode slots
  └── benchmark-facing stats

C++ / CUDA backend
  ├── Tensor / Storage / Device runtime
  ├── CUDA ops
  ├── Qwen2 end-to-end forward
  ├── TP=2 NCCL communication
  ├── Fixed-slot KV cache
  └── Paged KV cache + block manager
```

关键目录：


| 内容               | 路径                                |
| ---------------- | --------------------------------- |
| C API            | `include/llaisys/`，`src/llaisys/` |
| Qwen2 / Qwen2TP  | `src/models/qwen2/`               |
| CUDA 算子          | `src/ops/`                        |
| Python 封装        | `python/llaisys/`                 |
| Online benchmark | `test/bench_online_cb.py`         |
| TP 推理入口          | `test/tp_infer.py`                |
| 构建配置             | `xmake.lua`，`xmake/nvidia.lua`    |


---

## 环境依赖

推荐环境：

- Ubuntu 22.04
- NVIDIA Driver + CUDA Toolkit 12.x
- NCCL
- Xmake
- Python 3.12
- PyTorch CUDA wheel
- `transformers`
- `safetensors`
- `huggingface_hub`

如果 NCCL 来自 PyPI wheel，例如 `nvidia-nccl-cu12`，运行时可能需要设置：

```bash
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
```

---

## 编译安装

```bash
cd /path/to/llaisys

xmake f --nv-gpu=y -cv --root
xmake --root
xmake install --root

pip install ./python/
```

---

## 快速开始

### TP=2 分布式推理

```bash
python test/tp_infer.py \
  --model /root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-14B \
  --test \
  --device_ids 0,1
```

### Continuous Batching / Paged KV Benchmark

```bash
LLAISYS_PAGED_ATTN_V2=1 python test/bench_online_cb.py \
  --model /root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-14B \
  --device_ids 0,1 \
  --kv_mode paged \
  --paged_block_size 16 \
  --paged_max_blocks 4096 \
  --paged_prefill_scratch_slots 1 \
  --max_slots 32 \
  --arrival_rates inf \
  --num_requests 32 \
  --workload /home/dzy/za/tmp/real_workload_14b_p128d128.jsonl \
  --ignore_eos \
  --csv llaisys_tp2_p128d128.csv
```

---

## Benchmark 结果

### p128d128 Controlled Serving

测试设置：

```text
model = DeepSeek-R1-Distill-Qwen-14B
dtype = BF16
prompt_len = 128
output_len = 128
requests = 32
arrival = inf
ignore_eos = true
input = real ShareGPT token-id prompts
```

LLAISYS 结果：


| 系统      | 配置                              | Output tok/s | 说明                               |
| ------- | ------------------------------- | ------------ | -------------------------------- |
| LLAISYS | TP=1 Paged KV                   | 456          | 默认 paged attention               |
| LLAISYS | TP=1 Paged KV + Paged Attention | 474          | paged decode attention 特化 kernel |
| LLAISYS | TP=2 Paged KV                   | 632          | prefill workspace 后              |
| LLAISYS | TP=2 Paged KV + Paged Attention | 653          | 当前 controlled benchmark 最优结果     |


TP=2 最优配置的 serving 指标：


| 指标                          | 数值                |
| --------------------------- | ----------------- |
| Completed requests          | 32                |
| Wall time                   | 6.27 s            |
| QPS                         | 5.10 req/s        |
| Output throughput           | 653 output tok/s  |
| Total token throughput      | 1306 tok/s        |
| Avg TTFT                    | 1166 ms           |
| P90 / P99 TTFT              | 1867 ms / 2034 ms |
| Avg TPOT                    | 39.3 ms           |
| P99 latency                 | 6154 ms           |
| Peak memory                 | 23349 MiB         |
| Avg / Max active batch      | 31.7 / 32         |
| Prefill time                | 2.03 s            |
| Decode time                 | 4.15 s            |
| Decode step time            | 32.7 ms           |
| KV block size               | 16 tokens         |
| Peak used KV blocks         | 512               |
| KV capacity                 | 65536 tokens      |
| Used KV blocks after finish | 0                 |


### vLLM 对齐测试

为了对齐成熟 serving 框架，LLAISYS benchmark 支持使用相同 token-id prompt 调用 vLLM OpenAI serving。相同 p128d128、32 requests、max concurrency=32 输入下，vLLM 结果如下：


| 系统        | Output tok/s | Total tok/s | 平均 latency | p99 latency |
| --------- | ------------ | ----------- | ---------- | ----------- |
| vLLM TP=1 | 950          | 1899        | 4242 ms    | 4302 ms     |
| vLLM TP=2 | 1874         | 3747        | 2182 ms    | 2185 ms     |


LLAISYS 当前重点是实现并分析推理系统核心机制，包括 TP、Paged KV、Continuous Batching、prefill/decode profiling 与 serving benchmark。与 vLLM 的差距主要来自成熟框架中的高性能 paged attention、CUDA Graph bucket、varlen prefill 与 scheduler 策略。

---

## 核心实现

### TP=2 Tensor Parallel

LLAISYS 的 TP=2 路径采用 Megatron-style 张量并行：

- QKV / Gate-Up 使用列并行，每张卡计算部分输出通道。
- O-Proj / Down-Proj 使用行并行，每张卡计算部分输入贡献。
- 每层在 O-Proj / MLP Down 后执行 NCCL AllReduce 聚合 hidden states。
- 每个 rank 维护本地 KV cache，并在本地执行 GQA attention。
- Python 侧使用 Safetensors 按 rank 分片加载权重，再通过 C API 注册到 C++ 后端。

### Continuous Batching

Continuous Batching 用于 online serving 场景。每个请求占用一个 slot，并独立维护：

- `slot_id`
- `seq_len`
- `last_token`
- `finished`
- KV cache 位置

Scheduler 每轮收集 active slots，将它们组成 decode batch。请求完成后释放 slot，新请求可以继续进入，从而避免传统 lockstep batch 必须一起开始、一起结束的问题。

### Paged KV Cache / Block Manager

Fixed-slot KV 会为每个 slot 预留完整 `maxseq`：

```text
[max_slots, maxseq, n_kv_heads, head_dim]
```

Paged KV 将 KV cache 切成 token blocks：

```text
[num_blocks, block_size, n_kv_heads, head_dim]
```

每个请求通过 block table 记录逻辑 block 到物理 block 的映射：

```text
block_table[slot, logical_block] -> physical_block
```

这样请求只按实际 token 数申请 KV block，请求结束后释放 block 并复用，提升显存利用率与并发能力。

### Paged Decode Attention

Paged Decode Attention 在 decode 阶段直接通过 block table 从 paged KV cache 读取历史 K/V，避免将 KV gather 成连续 buffer。

当前实现包含：

- 默认 paged attention kernel
- block-table cache
- BR tile 调参
- paged attention 特化 kernel

Paged Attention 针对 `block_size=16`、`head_dim=128` 的 decode 场景做特化，按 paged block 扫描 KV，减少 position 到 physical block 的重复映射开销。

### Chunked Prefill + Decode Interleave

高压 online serving 中，如果先 prefill 所有请求，再进入 decode，会导致 TTFT 很高。LLAISYS 实现了 chunked prefill + decode interleave 实验路径：

- 将 prefill 按 token budget 分 chunk 推进。
- 每轮穿插 active requests 的 decode。
- 避免长 prompt prefill 长时间阻塞 decode。

### Profiling 与实验路径

LLAISYS 提供多种 profiling 与 A/B 开关：


| 开关                              | 作用                            |
| ------------------------------- | ----------------------------- |
| `LLAISYS_PREFILL_TIMING=1`      | 拆解 prefill 耗时                 |
| `LLAISYS_DECODE_TIMING=1`       | 拆解 decode 耗时                  |
| `LLAISYS_TP_TIMING=1`           | 拆解 NCCL AllReduce 耗时          |
| `LLAISYS_PAGED_ATTN_V2=1`       | 启用 paged attention 特化 kernel  |
| `LLAISYS_ENABLE_DECODE_GRAPH=1` | 尝试 decode CUDA Graph fallback |
| `LLAISYS_TP_OVERLAP=1`          | 启用 NCCL async allreduce 实验路径  |
| `LLAISYS_FUSED_QKV=1`           | 启用 fused QKV GEMM 实验路径        |
| `LLAISYS_FUSED_GATE_UP=1`       | 启用 fused GateUp GEMM 实验路径     |


---

## Roadmap

- 高性能 paged attention / GQA paged attention。
- Varlen prefill attention。
- CUDA Graph bucket for decode。
- 更成熟的 waiting/running scheduler。
- TP 通信与计算 overlap。
- Prefix cache / prefix sharing。
- KV cache quantization。

---

## License

本项目遵循仓库中的 LICENSE。