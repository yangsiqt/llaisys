import argparse
import ctypes
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import llaisys
from llaisys.models import Qwen2


DEFAULT_MODEL = "/model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


class ProfilerControl:
    def __init__(self):
        self._nvtx = ctypes.CDLL("libnvToolsExt.so")
        self._nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        self._nvtx.nvtxRangePushA.restype = ctypes.c_int
        self._nvtx.nvtxRangePop.argtypes = []
        self._nvtx.nvtxRangePop.restype = ctypes.c_int

        self._cudart = ctypes.CDLL("libcudart.so")
        self._cudart.cudaProfilerStart.argtypes = []
        self._cudart.cudaProfilerStart.restype = ctypes.c_int
        self._cudart.cudaProfilerStop.argtypes = []
        self._cudart.cudaProfilerStop.restype = ctypes.c_int
        self._cudart.cudaDeviceSynchronize.argtypes = []
        self._cudart.cudaDeviceSynchronize.restype = ctypes.c_int

    @contextmanager
    def range(self, name):
        self._nvtx.nvtxRangePushA(name.encode("utf-8"))
        try:
            yield
        finally:
            self._nvtx.nvtxRangePop()

    def synchronize(self):
        error = self._cudart.cudaDeviceSynchronize()
        if error:
            raise RuntimeError(f"cudaDeviceSynchronize failed with code {error}")

    def start(self):
        self.synchronize()
        error = self._cudart.cudaProfilerStart()
        if error:
            raise RuntimeError(f"cudaProfilerStart failed with code {error}")

    def stop(self):
        self.synchronize()
        error = self._cudart.cudaProfilerStop()
        if error:
            raise RuntimeError(f"cudaProfilerStop failed with code {error}")


def make_prompt(model, length, seed):
    rng = random.Random(seed)
    eos_token = int(model._config["eos_token_id"])
    high = min(int(model._config["vocab_size"]) - 1, 32000)
    tokens = [rng.randint(10, high) for _ in range(length)]
    return [10 if token == eos_token else token for token in tokens]


def run_sequence(model, prompt, output_length, profiler, annotate):
    model.reset_cache()
    tokens = list(prompt)
    timings = {"prefill_ms": 0.0, "decode_ms": 0.0}

    prefill_context = profiler.range("prefill") if annotate else _null_range()
    with prefill_context:
        start = time.perf_counter()
        tokens.append(model.infer_step(tokens))
        profiler.synchronize()
        timings["prefill_ms"] = (time.perf_counter() - start) * 1000.0

    decode_context = profiler.range("decode") if annotate else _null_range()
    with decode_context:
        start = time.perf_counter()
        for step in range(1, output_length):
            step_context = (
                profiler.range(f"decode_step_{step:03d}")
                if annotate
                else _null_range()
            )
            with step_context:
                tokens.append(model.infer_step(tokens))
        profiler.synchronize()
        timings["decode_ms"] = (time.perf_counter() - start) * 1000.0

    return tokens, timings


@contextmanager
def _null_range():
    yield


def main():
    parser = argparse.ArgumentParser(description="Profile single-GPU Qwen2 P128/D128 inference")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--warmup-output-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="profiles/run_metadata.json")
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Use cudaProfilerStart/Stop around the measured sequence",
    )
    args = parser.parse_args()

    if args.prompt_length <= 0 or args.output_length <= 0:
        parser.error("prompt and output lengths must be positive")

    profiler = ProfilerControl()
    with profiler.range("model_load"):
        model = Qwen2(args.model, llaisys.DeviceType.NVIDIA)
    prompt = make_prompt(model, args.prompt_length, args.seed)

    with profiler.range("warmup"):
        run_sequence(
            model,
            prompt,
            min(args.output_length, args.warmup_output_length),
            profiler,
            annotate=False,
        )

    if args.capture:
        profiler.start()
    try:
        with profiler.range("profile"):
            tokens, timings = run_sequence(
                model, prompt, args.output_length, profiler, annotate=True
            )
    finally:
        if args.capture:
            profiler.stop()

    decode_steps = max(0, args.output_length - 1)
    metadata = {
        "model": str(Path(args.model).resolve()),
        "dtype": model._config.get("torch_dtype"),
        "device": "nvidia:0",
        "prompt_length": args.prompt_length,
        "output_length": args.output_length,
        "decode_steps_after_prefill": decode_steps,
        "seed": args.seed,
        **timings,
        "decode_step_ms": timings["decode_ms"] / decode_steps if decode_steps else 0.0,
        "output_tokens": tokens[-args.output_length :],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k != "output_tokens"}, indent=2))


if __name__ == "__main__":
    main()
