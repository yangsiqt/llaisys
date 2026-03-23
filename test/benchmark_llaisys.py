#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一 benchmark：端到端（总耗时、token 速度）+ 可选算子级 profile。

说明
----
1) 端到端推理在 C++（llaisysQwen2ModelInfer / TP 路径）内完成，**不经过** Python ``Ops``，
   因此「各算子在一次真实 forward 里的耗时占比」需要改 C++ 打点；本脚本提供的算子数据来自
   ``test_ops.py`` 的**固定 shape** 微基准（与 ``python test_ops.py --profile`` 一致），
   用于对比 PyTorch 参考实现与 LLAISYS 算子，**不等价**于整网里的算子占比。

2) 单卡：``--mode single --device cpu|nvidia``
   双卡 TP：``--mode tp --device_ids 0,1``（需当前环境已安装并实现 ``Qwen2TP``）

用法
----
  cd /path/to/llaisys/test

  # 单卡 CPU，端到端 + 算子 profile
  python benchmark_llaisys.py --mode single --device cpu --model ../models/YourModel --profile-ops

  # 单卡 CUDA
  python benchmark_llaisys.py --mode single --device nvidia --model ../models/YourModel

  # 双卡 TP（你分支里有 Qwen2TP 时）
  python benchmark_llaisys.py --mode tp --device_ids 0,1 --model /path/to/14B

  # 仅算子 profile、不跑大模型
  python benchmark_llaisys.py --profile-ops --profile-device cpu --skip-e2e

  # 一行 JSON（仅端到端摘要）
  python benchmark_llaisys.py --mode single --device cpu --model ... --json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)


def _chat_inputs(tokenizer, prompt: str) -> List[int]:
    text = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(text)


def _run_e2e_single(
    model_path: str,
    device_name: str,
    prompt: str,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
) -> Dict[str, Any]:
    from test_utils import llaisys_device
    import llaisys
    from llaisys.models import Qwen2
    from transformers import AutoTokenizer

    if not os.path.isdir(model_path):
        raise SystemExit(f"--model 必须是本地目录: {model_path!r}")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    t_tok = time.perf_counter() - t0

    t1 = time.perf_counter()
    model = Qwen2(model_path, llaisys_device(device_name))
    t_load = time.perf_counter() - t1

    prompt_ids = _chat_inputs(tokenizer, prompt)
    n_prompt = len(prompt_ids)

    gc.collect()
    t2 = time.perf_counter()
    outputs = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    t_gen = time.perf_counter() - t2

    n_out = len(outputs)
    new_tokens = max(0, n_out - n_prompt)
    tps = (new_tokens / t_gen) if t_gen > 0 else 0.0
    ms_per_t = (t_gen / new_tokens * 1000.0) if new_tokens > 0 else None

    return {
        "backend": "single",
        "device": device_name,
        "model_path": os.path.abspath(model_path),
        "time_load_tokenizer_s": round(t_tok, 4),
        "time_load_model_s": round(t_load, 4),
        "time_generate_s": round(t_gen, 4),
        "time_total_after_start_s": round(t_tok + t_load + t_gen, 4),
        "tokens_prompt": n_prompt,
        "tokens_total": n_out,
        "tokens_new": new_tokens,
        "throughput_new_tokens_per_s": round(tps, 2),
        "latency_ms_per_new_token": round(ms_per_t, 3) if ms_per_t is not None else None,
    }


def _import_qwen2_tp():
    """兼容不同分支：Qwen2TP 可能在 models.__init__ 或 models.qwen2_tp。"""
    try:
        from llaisys.models import Qwen2TP

        return Qwen2TP
    except ImportError:
        pass
    try:
        import importlib

        m = importlib.import_module("llaisys.models.qwen2_tp")
        return m.Qwen2TP
    except Exception:
        return None


def _run_e2e_tp(
    model_path: str,
    device_ids: List[int],
    prompt: str,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
) -> Dict[str, Any]:
    import llaisys

    Qwen2TP = _import_qwen2_tp()
    if Qwen2TP is None:
        raise SystemExit(
            "当前 Python 包中未找到 Qwen2TP。"
            "请在你已实现张量并行的分支中导出 Qwen2TP，或改用 --mode single。"
        )

    from transformers import AutoTokenizer

    if not os.path.isdir(model_path):
        raise SystemExit(f"--model 必须是本地目录: {model_path!r}")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    t_tok = time.perf_counter() - t0

    t1 = time.perf_counter()
    model = Qwen2TP(model_path, llaisys.DeviceType.NVIDIA, device_ids=device_ids)
    t_load = time.perf_counter() - t1

    prompt_ids = _chat_inputs(tokenizer, prompt)
    n_prompt = len(prompt_ids)

    gc.collect()
    t2 = time.perf_counter()
    outputs = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    t_gen = time.perf_counter() - t2

    n_out = len(outputs)
    new_tokens = max(0, n_out - n_prompt)
    tps = (new_tokens / t_gen) if t_gen > 0 else 0.0
    ms_per_t = (t_gen / new_tokens * 1000.0) if new_tokens > 0 else None

    return {
        "backend": "tp",
        "device_ids": device_ids,
        "model_path": os.path.abspath(model_path),
        "time_load_tokenizer_s": round(t_tok, 4),
        "time_load_model_s": round(t_load, 4),
        "time_generate_s": round(t_gen, 4),
        "time_total_after_start_s": round(t_tok + t_load + t_gen, 4),
        "tokens_prompt": n_prompt,
        "tokens_total": n_out,
        "tokens_new": new_tokens,
        "throughput_new_tokens_per_s": round(tps, 2),
        "latency_ms_per_new_token": round(ms_per_t, 3) if ms_per_t is not None else None,
    }


def _run_ops_profile(device: str) -> None:
    from test_ops import run_all_tests

    run_all_tests(device=device, profile=True)


def _print_e2e(e: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("端到端 (E2E)")
    print("=" * 60)
    for k, v in e.items():
        print(f"  {k}: {v}")
    print(
        f"\n  近似 decode 吞吐 (仅统计新生成 token): "
        f"{e.get('throughput_new_tokens_per_s')} new-tokens/s"
    )
    lat = e.get("latency_ms_per_new_token")
    if lat is not None:
        print(f"  近似每新生成 token 延迟: {lat} ms/token")


def main() -> None:
    p = argparse.ArgumentParser(
        description="LLAISYS 统一 benchmark：E2E + 可选算子 profile（同 test_ops --profile）"
    )
    p.add_argument(
        "--mode",
        default="single",
        choices=["single", "tp"],
        help="single=Qwen2 单设备；tp=Qwen2TP 多卡（需包内实现）",
    )
    p.add_argument("--model", default=None, type=str, help="本地模型目录")
    p.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    p.add_argument(
        "--device_ids",
        default="0,1",
        type=str,
        help="TP 模式下的 GPU 编号列表，逗号分隔，例如 0,1",
    )
    p.add_argument("--prompt", default="Who are you?", type=str)
    p.add_argument("--max-steps", default=128, type=int, dest="max_steps")
    p.add_argument("--top_p", default=0.8, type=float)
    p.add_argument("--top_k", default=50, type=int)
    p.add_argument("--temperature", default=1.0, type=float)
    p.add_argument("--test", action="store_true", help="与 dzy_test_infer 一致：确定性采样参数")
    p.add_argument(
        "--profile-ops",
        action="store_true",
        help="运行与 test_ops.py --profile 相同的算子对比（Torch vs LLAISYS）",
    )
    p.add_argument(
        "--profile-device",
        default=None,
        choices=["cpu", "nvidia"],
        help="算子 profile 使用的设备（默认：single 与 --device 相同；tp 时默认 nvidia）",
    )
    p.add_argument("--skip-e2e", action="store_true", help="只跑算子 profile，不加载推理模型")
    p.add_argument("--json", action="store_true", help="仅将 E2E 结果打印为一行 JSON")

    args = p.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    result: Dict[str, Any] = {"e2e": None, "operator_profile_note": "见文件顶部 docstring"}

    if not args.skip_e2e:
        if not args.model:
            print("错误: 端到端需要 --model 指向本地模型目录", file=sys.stderr)
            sys.exit(2)
        if args.mode == "single":
            result["e2e"] = _run_e2e_single(
                args.model,
                args.device,
                args.prompt,
                args.max_steps,
                top_p,
                top_k,
                temperature,
            )
        else:
            ids = [int(x.strip()) for x in args.device_ids.split(",") if x.strip() != ""]
            result["e2e"] = _run_e2e_tp(
                args.model,
                ids,
                args.prompt,
                args.max_steps,
                top_p,
                top_k,
                temperature,
            )

        if args.json:
            print(json.dumps(result["e2e"], ensure_ascii=False))
        else:
            _print_e2e(result["e2e"])

    if args.profile_ops:
        prof_dev = args.profile_device
        if prof_dev is None:
            prof_dev = "nvidia" if args.mode == "tp" else args.device
        if not args.json:
            print("\n" + "=" * 60)
            print(f"算子级 profile（device={prof_dev}，与 test_ops.py --profile 相同）")
            print("=" * 60)
        _run_ops_profile(prof_dev)

    if args.skip_e2e and not args.profile_ops:
        print("请指定 --profile-ops，或去掉 --skip-e2e 并提供 --model", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
