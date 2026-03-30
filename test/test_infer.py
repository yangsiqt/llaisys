import gc
import io
import subprocess
import sys
import threading
import time
from ctypes import c_int64
from dataclasses import dataclass
from test_utils import *

import argparse
import os
import torch
from huggingface_hub import snapshot_download
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.models import Qwen2
from transformers import AutoModelForCausalLM, AutoTokenizer

import llaisys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


@dataclass
class InferStats:
    n_prompt: int
    n_decode: int
    time_total_s: float
    time_prefill_s: float
    time_decode_s: float
    peak_smi_mib: int

    @property
    def p_tok_s(self) -> float:
        return self.n_prompt / self.time_prefill_s if self.time_prefill_s > 0 else float("inf")

    @property
    def d_tok_s(self) -> float:
        return self.n_decode / self.time_decode_s if self.time_decode_s > 0 else float("nan")


class GpuMemSampler:
    """后台轮询 nvidia-smi，记录采样期间 memory.used 峰值（MiB）。"""

    def __init__(self, gpu_index: int = 0):
        self._gpu_index = gpu_index
        self._peak_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_loop(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(self._gpu_index),
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                if out.returncode == 0 and out.stdout.strip():
                    v = int(out.stdout.strip().split("\n")[0])
                    self._peak_mib = max(self._peak_mib, v)
            except (subprocess.TimeoutExpired, ValueError, IndexError, FileNotFoundError):
                pass
            time.sleep(0.02)

    def start(self):
        self._peak_mib = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        return self._peak_mib


def load_hf_model(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path


def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs[0].tolist(), result


def hf_infer_profiled(
    prompt,
    tokenizer,
    model,
    max_new_tokens=128,
    top_p=0.8,
    top_k=50,
    temperature=0.8,
    sampler: GpuMemSampler | None = None,
) -> tuple[list[int], str, InferStats]:
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    n_prompt = int(inputs.shape[1])

    if model.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(inputs.device)

    if sampler is not None:
        sampler.start()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    peak_smi = sampler.stop() if sampler is not None else 0

    flat = outputs[0].tolist()
    n_decode = len(flat) - n_prompt
    # HuggingFace generate 为单段计时，prefill/decode 不拆分；p/d 用全程折算占位便于对比展示
    total_t = t1 - t0
    stats = InferStats(
        n_prompt=n_prompt,
        n_decode=n_decode,
        time_total_s=total_t,
        time_prefill_s=total_t * (n_prompt / max(n_prompt + n_decode, 1)),
        time_decode_s=total_t * (n_decode / max(n_prompt + n_decode, 1)),
        peak_smi_mib=peak_smi,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return flat, result, stats


def load_llaisys_model(model_path, device_name) -> Qwen2:
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    return model


def llaisys_generate_timed(
    model: Qwen2,
    inputs: list[int],
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    sampler: GpuMemSampler | None = None,
) -> tuple[list[int], InferStats]:
    """与 Qwen2.generate 逻辑一致，分别累计首包 prefill 与逐 token decode 耗时。"""
    if top_k != 1:
        print("Warning: Only greedy sampling (top_k=1) is currently supported")

    output_tokens = list(inputs)
    n_prompt = len(inputs)
    t_prefill = 0.0
    t_decode = 0.0
    first = True

    if sampler is not None:
        sampler.start()

    t_total0 = time.perf_counter()
    for _i in range(max_new_tokens):
        token_array = (c_int64 * len(output_tokens))(*output_tokens)
        t0 = time.perf_counter()
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            model._model, token_array, len(output_tokens)
        )
        t1 = time.perf_counter()
        dt = t1 - t0
        if first:
            t_prefill += dt
            first = False
        else:
            t_decode += dt

        if next_token == model._config["eos_token_id"]:
            output_tokens.append(int(next_token))
            break
        output_tokens.append(int(next_token))

    t_total1 = time.perf_counter()
    peak_smi = sampler.stop() if sampler is not None else 0

    n_decode = len(output_tokens) - n_prompt
    stats = InferStats(
        n_prompt=n_prompt,
        n_decode=n_decode,
        time_total_s=t_total1 - t_total0,
        time_prefill_s=t_prefill,
        time_decode_s=t_decode,
        peak_smi_mib=peak_smi,
    )
    return output_tokens, stats


def llaisys_infer_profiled(
    prompt,
    tokenizer,
    model: Qwen2,
    max_new_tokens=128,
    top_p=0.8,
    top_k=50,
    temperature=0.8,
    sampler: GpuMemSampler | None = None,
) -> tuple[list[int], str, InferStats]:
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    output_tokens, stats = llaisys_generate_timed(
        model,
        inputs,
        max_new_tokens,
        top_k,
        top_p,
        temperature,
        sampler=sampler,
    )
    text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return output_tokens, text, stats


def print_benchmark_footer(
    device_name: str,
    hf_stats: InferStats | None,
    hf_torch_peak_mib: float | None,
    llaisys_stats: InferStats | None,
):
    print("\n=== 显存与推理速度 ===\n")
    if device_name != "nvidia":
        print("当前非 --device nvidia，未采集 GPU 峰值显存（nvidia-smi）。")
        if llaisys_stats:
            print(
                f"[LLAISYS] prompt={llaisys_stats.n_prompt} tokens, "
                f"decode={llaisys_stats.n_decode} tokens, 总用时 {llaisys_stats.time_total_s:.3f}s"
            )
            print(
                f"  prefill: p = {llaisys_stats.n_prompt / llaisys_stats.time_prefill_s:.2f} tok/s "
                f"({llaisys_stats.time_prefill_s:.4f}s)"
            )
            if llaisys_stats.n_decode > 0:
                print(
                    f"  decode:  d = {llaisys_stats.n_decode / llaisys_stats.time_decode_s:.2f} tok/s "
                    f"({llaisys_stats.time_decode_s:.4f}s)"
                )
            else:
                print("  decode:  d = N/A（无新生成 token）")
        return

    if hf_stats is not None:
        tot = hf_stats.n_prompt + hf_stats.n_decode
        overall = tot / hf_stats.time_total_s if hf_stats.time_total_s > 0 else 0.0
        print("[HuggingFace 参考]")
        print(f"  nvidia-smi 采样峰值显存: {hf_stats.peak_smi_mib} MiB")
        if hf_torch_peak_mib is not None:
            print(f"  PyTorch peak reserved:     {hf_torch_peak_mib:.1f} MiB")
        print(
            f"  prompt {hf_stats.n_prompt} + 生成 {hf_stats.n_decode} tokens, "
            f"总用时 {hf_stats.time_total_s:.3f}s"
        )
        print(
            f"  全程平均（prefill+decode）: {overall:.2f} tok/s "
            f"（HF 单次 generate 未拆分 p/d，此为整体）"
        )
        print()

    if llaisys_stats is not None:
        print("[LLAISYS]")
        print(f"  nvidia-smi 采样峰值显存: {llaisys_stats.peak_smi_mib} MiB")
        print(
            f"  prompt {llaisys_stats.n_prompt} + 生成 {llaisys_stats.n_decode} tokens, "
            f"总用时 {llaisys_stats.time_total_s:.3f}s"
        )
        p_rate = (
            llaisys_stats.n_prompt / llaisys_stats.time_prefill_s
            if llaisys_stats.time_prefill_s > 0
            else float("inf")
        )
        print(
            f"  prefill: p = {p_rate:.2f} tok/s "
            f"（{llaisys_stats.n_prompt} tokens / {llaisys_stats.time_prefill_s:.4f}s）"
        )
        if llaisys_stats.n_decode > 0 and llaisys_stats.time_decode_s > 0:
            d_rate = llaisys_stats.n_decode / llaisys_stats.time_decode_s
            print(
                f"  decode:  d = {d_rate:.2f} tok/s "
                f"（{llaisys_stats.n_decode} tokens / {llaisys_stats.time_decode_s:.4f}s）"
            )
        else:
            print("  decode:  d = N/A（无 decode 步或耗时为 0）")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    tokenizer, model, model_path = load_hf_model(args.model, args.device)

    hf_sampler = GpuMemSampler(0) if args.device == "nvidia" else None
    tokens, output, hf_stats = hf_infer_profiled(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        sampler=hf_sampler,
    )
    hf_torch_peak_mib = None
    if args.device == "nvidia" and model.device.type == "cuda":
        hf_torch_peak_mib = torch.cuda.max_memory_reserved(model.device) / (1024 * 1024)

    del model
    gc.collect()
    if args.device == "nvidia":
        torch.cuda.empty_cache()

    print("\n=== Answer ===\n")
    print("Tokens:")
    print(tokens)
    print("\nContents:")
    print(output)
    print("\n")
    print(f"Time elapsed: {hf_stats.time_total_s:.2f}s\n")

    llaisys_model = load_llaisys_model(model_path, args.device)
    ls_sampler = GpuMemSampler(0) if args.device == "nvidia" else None
    llaisys_tokens, llaisys_output, llaisys_stats = llaisys_infer_profiled(
        args.prompt,
        tokenizer,
        llaisys_model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        sampler=ls_sampler,
    )

    print("\n=== Your Result ===\n")
    print("Tokens:")
    print(llaisys_tokens)
    print("\nContents:")
    print(llaisys_output)
    print("\n")
    print(f"Time elapsed: {llaisys_stats.time_total_s:.2f}s\n")

    if args.test:
        assert llaisys_tokens == tokens
        print("\033[92mTest passed!\033[0m\n")

    print_benchmark_footer(
        args.device,
        hf_stats if args.device == "nvidia" else None,
        hf_torch_peak_mib,
        llaisys_stats,
    )
