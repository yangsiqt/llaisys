import gc
from test_utils import *

import argparse
from transformers import AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
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


def llaisys_infer(
    prompt,
    tokenizer,
    model: Qwen2TP,
    max_new_tokens=128,
    top_p=0.8,
    top_k=50,
    temperature=0.8,
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--device_ids", default="0,1", type=str,
                        help="Comma-separated GPU device IDs for tensor parallelism")

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

    start_time = time.time()
    llaisys_tokens, llaisys_output = llaisys_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )

    end_time = time.time()

    print("\n=== TP Inference Result ===\n")
    print("Tokens:")
    print(llaisys_tokens)
    print("\nContents:")
    print(llaisys_output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s")
    print(f"Tokens generated: {len(llaisys_tokens)}")
    tok_per_sec = len(llaisys_tokens) / (end_time - start_time) if (end_time - start_time) > 0 else 0
    print(f"Speed: {tok_per_sec:.2f} tokens/s\n")

    if args.test:
        print("\033[92mTP inference completed successfully.\033[0m\n")
