from test_utils import *

import argparse
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import os
import subprocess
import threading
import time
import llaisys
from llaisys.models import Qwen2TP
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def load_tokenizer(model_path=None):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    if model_path and os.path.isdir(model_path):
        print(f"Loading tokenizer from local path: {model_path}")
    else:
        print(f"Downloading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer, model_path


def load_llaisys_tp_model(model_path, device_ids) -> Qwen2TP:
    model = Qwen2TP(model_path, llaisys.DeviceType.NVIDIA, device_ids=device_ids)
    return model


def query_gpu_mem_mib_all():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    out = {}
    for line in r.stdout.strip().split("\n"):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            out[int(parts[0])] = int(parts[1])
    return out


class GpuMemPeakSampler:
    """后台轮询 nvidia-smi，记录指定 GPU 的 memory.used 峰值（MiB）。"""

    def __init__(self, device_ids, interval_s=0.05):
        self.device_ids = list(device_ids)
        self.interval_s = interval_s
        self.peaks = {i: 0 for i in self.device_ids}
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            try:
                cur = query_gpu_mem_mib_all()
                for i in self.device_ids:
                    self.peaks[i] = max(self.peaks[i], cur.get(i, 0))
            except Exception:
                pass

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


def print_llaisys_benchmark_footer(device_ids, metrics: dict):
    n_p = metrics["n_prompt"]
    n_g = metrics["n_generated"]
    prefill_s = metrics["prefill_s"]
    decode_wall_s = metrics["decode_wall_s"]
    total_s = metrics["total_s"]
    sum_dec = metrics["sum_decode_kernel_s"]

    p_tok_s = (n_p / prefill_s) if prefill_s > 0 else 0.0
    d_tok_s = (n_g / decode_wall_s) if decode_wall_s > 0 and n_g > 0 else 0.0
    d_kernel_tok_s = (max(0, n_g - 1) / sum_dec) if sum_dec > 0 and n_g > 1 else 0.0

    peaks = metrics.get("peak_mem_mib", {})
    mem_parts = [f"GPU{i} 峰值 {peaks.get(i, 0)} MiB" for i in device_ids]
    mem_line = "  nvidia-smi 采样峰值显存: " + " | ".join(mem_parts)

    print("\n[LLAISYS]")
    print(mem_line)
    print(f"  prompt {n_p} + 生成 {n_g} tokens, 总用时 {total_s:.3f}s")
    print(f"  prefill: p = {p_tok_s:.2f} tok/s （{n_p} tokens / {prefill_s:.4f}s）")
    print(f"  decode:  d = {d_tok_s:.2f} tok/s （{n_g} tokens / {decode_wall_s:.4f}s）")
    if sum_dec > 0 and n_g > 1:
        print(
            f"  decode(仅内核累计): {d_kernel_tok_s:.2f} tok/s "
            f"（{n_g - 1} steps / {sum_dec:.4f}s）"
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--device_ids",
        default="0,1",
        type=str,
        help="Comma-separated GPU device IDs for tensor parallelism",
    )

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    device_ids = [int(x) for x in args.device_ids.split(",")]
    print(f"Tensor Parallel with device_ids={device_ids}")

    tokenizer, model_path = load_tokenizer(args.model)

    print(f"Model path: {model_path}")
    sys.stdout.flush()

    model = load_llaisys_tp_model(model_path, device_ids)

    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = tokenizer.encode(input_content)

    sampler = GpuMemPeakSampler(device_ids)
    sampler.start()
    try:
        llaisys_tokens, metrics = model.generate_with_pd_metrics(
            prompt_ids,
            max_new_tokens=args.max_steps,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
    finally:
        sampler.stop()

    try:
        cur = query_gpu_mem_mib_all()
        for i in device_ids:
            sampler.peaks[i] = max(sampler.peaks[i], cur.get(i, 0))
    except Exception:
        pass
    metrics["peak_mem_mib"] = dict(sampler.peaks)
    llaisys_output = tokenizer.decode(llaisys_tokens, skip_special_tokens=True)

    print("\n=== TP Inference Result ===\n")
    print("Tokens:")
    print(llaisys_tokens)
    print("\nContents:")
    print(llaisys_output)
    print("\n")

    print_llaisys_benchmark_footer(device_ids, metrics)

    if args.test:
        print("\033[92mTP inference completed successfully.\033[0m\n")
