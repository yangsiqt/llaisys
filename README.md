# LLM 推理系统：张量 → 算子 → 端到端 →CPU/GPU/分布式

本仓库用 **多个 Git 分支** 分别承载 **CPU 优化**、**单卡 GPU（CUTLASS + cuBLAS）** 与 **双卡张量并行（NCCL TP）**；编译与入口脚本相同，但 **请先 `git checkout` 到对应分支** 再构建，避免与代码不一致。本工程使用Python 负责编排与 HF 生态对接，经 C 接口调用 C++/CUDA；算子与推理下沉原生层，兼顾迭代效率与算力。

---

## 1. 概述

### 1.1 分支功能

| 分支 | 侧重点 | 典型场景 |
|------|--------|----------|
| **`feature/cpu`** | **OpenBLAS** + **OpenMP** + **AVX-512** 等 CPU 编译与 `linear` 等算子优化（`LLAISYS_USE_OPENBLAS` 等，见 `xmake/cpu.lua`） | 1.5B **纯 CPU** 端到端、`test/dzy_test_infer.py --device cpu` |
| **`perf/cutlass`** | 引入 **`third_party/cutlass`** 头文件，NVIDIA `linear` 等与 **cuBLAS / CUTLASS** 对齐的单卡 GPU 路径；`xmake/nvidia.lua` 以 **sm_86** 等为主（适合 **RTX 3090** 一类） | 1.5B **单卡 GPU**、`dzy_test_infer.py --device nvidia` |
| **`feature/tp`** | 在 GPU 算子线路上叠加 **NCCL**、**Megatron 式张量并行**、`Qwen2TP`、`test/tp_infer.py`；`xmake/nvidia.lua` 含 **NCCL** 与多架构（如 **sm_80、sm_86**，便于 **A800** 等） | **DeepSeek-R1-Distill-Qwen-14B**、**TP=2** 双卡 |

### 1.2 核心代码

| 内容 | 位置 |
|------|------|
| C API / 模型 | `include/llaisys/`，`src/llaisys/`，`src/models/` 等 |
| Python 封装 | `python/llaisys/`（`Qwen2`；**`Qwen2TP` 在 `feature/tp`**） |
| 1.5B 端到端（CPU 或单卡 GPU） | `test/dzy_test_infer.py` |
| 14B 双卡 TP | **`feature/tp`**：`test/tp_infer.py` |
| 构建 | `xmake.lua`，`xmake/cpu.lua`，`xmake/nvidia.lua` |

---

## 2. 环境安装与配置

### 2.1 通用依赖

- **系统**：Ubuntu 22.04  
- **构建**：[Xmake](https://xmake.io/)  
- **编译器**：GCC 或 Clang（C++17）  
- **Python**：3.12.3（conda base 实测）  
- **Python 包**：`torch` 2.8.0+cu128，`transformers` 5.2.0，`huggingface_hub` 1.5.0，`safetensors` 0.7.0  
- **CUDA**：12.8（与上述 `torch` wheel 一致）

### 2.2 CPU 路径（`feature/cpu`）

- 安装 **OpenBLAS**，并保证链接期能找到库；`feature/cpu` 下 `xmake/cpu.lua` 默认示例为 Ubuntu 常见路径（`/usr/lib/x86_64-linux-gnu` 等），若路径不同需自行改 `xmake/cpu.lua` 或做软链接。  
- 运行时用 **`OMP_NUM_THREADS`**、**`OPENBLAS_NUM_THREADS`** 与机器核心数对齐。

### 2.3 GPU 路径（`perf/cutlass` / `feature/tp`）

- **CUDA Toolkit**、驱动；**CUTLASS**：在含 `third_party/cutlass` 的分支上，首次克隆后需拉取子模块（见第 3 节）。  
- **NCCL（仅 `feature/tp`）**：需开发头文件与动态库；若使用 PyPI 的 **`nvidia-nccl-cu12`** 等 wheel，运行时常需将 **`…/site-packages/nvidia/nccl/lib`** 加入 **`LD_LIBRARY_PATH`**。  
- `feature/tp` 的 `xmake/nvidia.lua` 会尝试探测 conda 下 `nvidia/nccl`，否则回退 **`/usr/include`** 与 **`/usr/lib/x86_64-linux-gnu`**，请与机器实际安装一致。

### 2.4 模型下载（可选镜像）

```bash
mkdir -p /path/to/models
export HF_ENDPOINT=https://hf-mirror.com
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    local_dir='/path/to/models/DeepSeek-R1-Distill-Qwen-14B',
    resume_download=True,
)
"
```

1.5B模型 将 repo id 换为 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` 即可。

---

## 3. 编译与安装

**务必先切换到目标分支**，再在仓库根目录执行。

### 3.1 开启 CUDA（`perf/cutlass` 或 `feature/tp`）

```bash
cd /path/to/llaisys
git checkout perf/cutlass   # 或: git checkout feature/tp
xmake f --nv-gpu=y -cv --root
xmake --root
xmake install --root
pip install ./python/
```

### 3.2 仅 CPU（`feature/cpu`）

```bash
cd /path/to/llaisys
git checkout feature/cpu
xmake f -c
xmake f --nv-gpu=n
xmake --root
xmake install --root
pip install ./python/
```

---

## 4. 运行

### 4.1 CPU：DeepSeek-R1-Distill-Qwen-1.5B（`feature/cpu`）

`--test` 下脚本会将采样设为确定性配置（见 `test/dzy_test_infer.py`）。

```bash
git checkout feature/cpu
# … 按 3.2 节完成仅 CPU 构建与 pip install …

cd /path/to/llaisys
python test/dzy_test_infer.py --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B/ --test --device cpu
```

### 4.2 单卡 GPU（RTX 3090 等）：同一 1.5B 脚本（`perf/cutlass` 或 `feature/tp`）

使用 CUDA 构建后，指定 **`--device nvidia`**（模型路径按本机修改，下例为相对目录）：

```bash
git checkout perf/cutlass   # 或 feature/tp（单卡可不跑 TP，仅跑 Qwen2）
# … 按 3.1 节完成 CUDA 构建与 pip install …

cd /path/to/llaisys
python test/dzy_test_infer.py --model models/DeepSeek-R1-Distill-Qwen-1.5B/ --test --device nvidia
```

### 4.3 双卡张量并行：14B + `test/tp_infer.py`（**仅 `feature/tp`**）

环境示例：**2×A800 NVLink**，模型 **DeepSeek-R1-Distill-Qwen-14B**，**TP=2**。需保证运行时能加载 **NCCL**：

```bash
git checkout feature/tp
# … 按 3.1 节完成 CUDA + NCCL 构建与 pip install …

cd /home/dzy/za/llaisys
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
python test/tp_infer.py \
  --model /root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-14B \
  --test \
  --device_ids 0,1
```

常用参数见 `test/tp_infer.py`：`--prompt`、`--max_steps`、`--device_ids`。脚本会输出 **prefill / decode 速度、**decode（仅内核累计）** 及 **nvidia-smi 轮询峰值显存**。

---

## 5. 结果与性能

### 5.1 CPU：1.5B（`feature/cpu`，128 tokens 量级）

**配置**：模型 **DeepSeek-R1-Distill-Qwen-1.5B**，分支 **`feature/cpu`**，`--test`，`--device cpu`；运行方式见 **§4.1**。  
**硬件参考**：Intel Xeon Platinum 8358P @ 2.60GHz，**15 vCPU**，内存约 90GB。

| 指标 | 优化前 | 优化后 | 提升（约） |
|------|---------------------|--------|------------|
| 端到端生成耗时 | ~514 s | **~7.4–7.9 s** | **~65×–71×** |
| 每 token 平均延时（同口径） | ~4.0 s | ~0.06 s | 同量级 |

### 5.2 GPU：单卡 LLAISYS vs HuggingFace BF16（**RTX 3090**）

**配置**：模型 **DeepSeek-R1-Distill-Qwen-1.5B**，分支 **`perf/cutlass` 或 `feature/tp`**（单卡），`--test`，`--device nvidia`，`test/dzy_test_infer.py`；运行方式见 **§4.2**。

| 项 | LLAISYS | HuggingFace BF16 参考 |
|----|---------|------------------------|
| 端到端耗时（同任务设定） | **~0.8 s** | ~3.2 s |
| 峰值显存（LLAISYS 侧） | **~7.5 GB** | — |
| 吞吐 | prefill **~692 tok/s**，decode **~98 tok/s** | — |

### 5.3 双卡 TP：14B（**2×A800 NVLink**，`feature/tp`，`test/tp_infer.py` **PD 统计**）

**配置**：DeepSeek-R1-Distill-Qwen-14B，**TP=2**，`--test`；命令见 **§4.3**。

| 指标 | 数值（本次实测） |
|------|------------------|
| **Prefill** | **~25.7 tok/s**（**9** 个 prompt token，约 **0.35 s**） |
| **Decode** | **~22.6 tok/s**（**81** 个生成 token，约 **3.59 s**） |
| **Decode** | **~22.3 tok/s**（**80** steps，约 **3.59 s**） |
| **显存峰值** | **GPU0 ~17125 MiB**，**GPU1 ~15639 MiB** |

---

