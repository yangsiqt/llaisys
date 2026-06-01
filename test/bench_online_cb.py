import argparse
import csv
import json
import random
import subprocess
import threading
import time
from pathlib import Path

import llaisys
from llaisys.models import Qwen2TP, Qwen2TPContinuousEngine


def parse_int_list(value):
    return [int(x) for x in value.split(",") if x.strip()]


def parse_float_list(value):
    return [float(x) for x in value.split(",") if x.strip()]


def query_gpu_mem_mib_all():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    out = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            out[int(parts[0])] = int(parts[1])
    return out


class GpuMemPeakSampler:
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


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def make_prompt_tokens(length, vocab_size, eos_token_id, rng):
    lo = 10
    hi = max(lo + 1, min(vocab_size - 1, 32000))
    tokens = [rng.randint(lo, hi) for _ in range(length)]
    if eos_token_id in tokens:
        tokens = [lo if t == eos_token_id else t for t in tokens]
    return tokens


def build_requests(arrival_rate, duration_s, prompt_lens, output_lens, vocab_size, eos_token_id, rng):
    requests = []
    t = 0.0
    req_id = 0
    interval = 1.0 / arrival_rate
    while t < duration_s:
        prompt_len = rng.choice(prompt_lens)
        output_len = rng.choice(output_lens)
        requests.append(
            {
                "request_id": req_id,
                "arrival_s": t,
                "prompt_tokens": make_prompt_tokens(prompt_len, vocab_size, eos_token_id, rng),
                "prompt_len": prompt_len,
                "output_len": output_len,
            }
        )
        req_id += 1
        t += interval
    return requests


def load_workload(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "prompt_tokens" not in rec:
                raise ValueError("workload records must include prompt_tokens")
            rec["prompt_tokens"] = [int(x) for x in rec["prompt_tokens"]]
            rec["prompt_len"] = int(rec.get("prompt_len", len(rec["prompt_tokens"])))
            rec["output_len"] = int(rec["output_len"])
            records.append(rec)
    if not records:
        raise ValueError(f"empty workload: {path}")
    return records


def assign_arrivals(records, arrival_rate, duration_s, rng, pattern):
    requests = []
    t = 0.0
    interval = 1.0 / arrival_rate
    for i, rec in enumerate(records):
        if t >= duration_s:
            break
        req = {
            "request_id": int(rec.get("request_id", i)),
            "arrival_s": t,
            "prompt_tokens": rec["prompt_tokens"],
            "prompt_len": rec["prompt_len"],
            "output_len": rec["output_len"],
        }
        requests.append(req)
        if pattern == "poisson":
            t += rng.expovariate(arrival_rate)
        else:
            t += interval
    return requests


def run_one(model, args, arrival_rate, device_ids):
    rng = random.Random(args.seed + int(arrival_rate * 1000))
    eos_token_id = model._config.get("eos_token_id")
    vocab_size = model._config["vocab_size"]
    if args.workload:
        records = load_workload(args.workload)
        rng.shuffle(records)
        if args.num_requests > 0:
            records = records[: args.num_requests]
        requests = assign_arrivals(records, arrival_rate, args.duration, rng, args.arrival_pattern)
    else:
        requests = build_requests(
            arrival_rate,
            args.duration,
            parse_int_list(args.prompt_lens),
            parse_int_list(args.output_lens),
            vocab_size,
            eos_token_id,
            rng,
        )

    engine = Qwen2TPContinuousEngine(
        model,
        max_slots=args.max_slots,
        eos_token_id=eos_token_id,
        ignore_eos=args.ignore_eos,
    )
    sampler = GpuMemPeakSampler(device_ids)
    waiting = []
    completed = []
    active_samples = []
    req_idx = 0

    start = time.perf_counter()
    sampler.start()
    try:
        while req_idx < len(requests) or waiting or engine.active_count() > 0:
            now = time.perf_counter()
            elapsed = now - start

            while req_idx < len(requests) and requests[req_idx]["arrival_s"] <= elapsed:
                req = requests[req_idx]
                req["arrival_abs_s"] = start + req["arrival_s"]
                waiting.append(req)
                req_idx += 1

            while waiting and engine.can_accept():
                if args.prefill_wait_ms > 0 and req_idx < len(requests):
                    oldest_wait_s = time.perf_counter() - waiting[0]["arrival_abs_s"]
                    free_slots = len(engine.free_slots)
                    if oldest_wait_s * 1000.0 < args.prefill_wait_ms and len(waiting) < free_slots:
                        break
                free_slots = len(engine.free_slots)
                first_len = waiting[0]["prompt_len"]
                batch = []
                keep = []
                for req in waiting:
                    if req["prompt_len"] == first_len and len(batch) < free_slots:
                        batch.append(req)
                    else:
                        keep.append(req)
                waiting = keep
                if len(batch) == 1:
                    req = batch[0]
                    done = engine.submit(
                        req["request_id"],
                        req["prompt_tokens"],
                        req["output_len"],
                        req["arrival_abs_s"],
                    )
                    if done is not None:
                        completed.append(done)
                else:
                    completed.extend(engine.submit_many_same_len(batch, time.perf_counter()))

            if engine.active_count() > 0:
                completed.extend(engine.step(time.perf_counter()))
                active_samples.append(engine.active_count())
            elif req_idx < len(requests):
                sleep_s = max(0.0, start + requests[req_idx]["arrival_s"] - time.perf_counter())
                time.sleep(min(sleep_s, 0.001))
    finally:
        sampler.stop()

    wall_s = time.perf_counter() - start
    prompt_tokens = sum(c["prompt_len"] for c in completed)
    output_tokens = sum(len(c["output_tokens"]) for c in completed)
    prompt_lens_completed = [c["prompt_len"] for c in completed]
    output_lens_completed = [len(c["output_tokens"]) for c in completed]
    ttft_ms = [(c["first_token_s"] - c["arrival_s"]) * 1000.0 for c in completed]
    latency_ms = [(c["finish_s"] - c["arrival_s"]) * 1000.0 for c in completed]
    tpot_ms = []
    for c in completed:
        n = len(c["output_tokens"])
        if n > 1:
            tpot_ms.append(((c["finish_s"] - c["first_token_s"]) / (n - 1)) * 1000.0)

    peak_mem = 0
    try:
        cur = query_gpu_mem_mib_all()
        for i in device_ids:
            sampler.peaks[i] = max(sampler.peaks[i], cur.get(i, 0))
    except Exception:
        pass
    for i in device_ids:
        peak_mem = max(peak_mem, sampler.peaks.get(i, 0))

    engine_metrics = engine.metrics()
    prefill_s = engine_metrics["prefill_s"]
    decode_s = engine_metrics["decode_s"]
    prefill_tok_s = engine_metrics["prefill_tokens"] / prefill_s if prefill_s > 0 else 0.0
    decode_slot_steps = engine_metrics["decode_slot_steps"]
    decode_step_s = decode_s / engine_metrics["decode_steps"] if engine_metrics["decode_steps"] > 0 else 0.0
    decode_slot_step_s = decode_s / decode_slot_steps if decode_slot_steps > 0 else 0.0
    prefill_batch_avg = (
        engine_metrics["prefill_batches"] / engine_metrics["prefill_calls"]
        if engine_metrics["prefill_calls"] > 0
        else 0.0
    )

    return {
        "arrival_rate": arrival_rate,
        "qps": len(completed) / wall_s if wall_s > 0 else 0.0,
        "completed": len(completed),
        "output_tok_s": output_tokens / wall_s if wall_s > 0 else 0.0,
        "total_tok_s": (prompt_tokens + output_tokens) / wall_s if wall_s > 0 else 0.0,
        "avg_ttft_ms": sum(ttft_ms) / len(ttft_ms) if ttft_ms else 0.0,
        "p50_ttft_ms": percentile(ttft_ms, 50),
        "p90_ttft_ms": percentile(ttft_ms, 90),
        "p99_ttft_ms": percentile(ttft_ms, 99),
        "avg_tpot_ms": sum(tpot_ms) / len(tpot_ms) if tpot_ms else 0.0,
        "p99_latency_ms": percentile(latency_ms, 99),
        "peak_mem_mib": peak_mem,
        "avg_active": sum(active_samples) / len(active_samples) if active_samples else 0.0,
        "max_active": max(active_samples) if active_samples else 0,
        "wall_s": wall_s,
        "prompt_len_avg": sum(prompt_lens_completed) / len(prompt_lens_completed) if prompt_lens_completed else 0.0,
        "prompt_len_p90": percentile(prompt_lens_completed, 90),
        "output_len_avg": sum(output_lens_completed) / len(output_lens_completed) if output_lens_completed else 0.0,
        "prefill_calls": engine_metrics["prefill_calls"],
        "prefill_batch_avg": prefill_batch_avg,
        "prefill_s": prefill_s,
        "prefill_tok_s": prefill_tok_s,
        "decode_steps": engine_metrics["decode_steps"],
        "decode_s": decode_s,
        "decode_step_ms": decode_step_s * 1000.0,
        "decode_slot_step_ms": decode_slot_step_s * 1000.0,
    }


def print_summary(row):
    print(
        f"arrival={row['arrival_rate']:.2f} req/s | completed={row['completed']} | "
        f"qps={row['qps']:.2f} | output={row['output_tok_s']:.2f} tok/s | "
        f"TTFT avg/p90/p99={row['avg_ttft_ms']:.1f}/{row['p90_ttft_ms']:.1f}/{row['p99_ttft_ms']:.1f} ms | "
        f"TPOT={row['avg_tpot_ms']:.1f} ms | p99 latency={row['p99_latency_ms']:.1f} ms | "
        f"active avg/max={row['avg_active']:.1f}/{row['max_active']} | mem={row['peak_mem_mib']} MiB"
    )
    print(
        f"  profile: wall={row['wall_s']:.2f}s | prompt avg/p90={row['prompt_len_avg']:.1f}/{row['prompt_len_p90']:.0f} | "
        f"prefill calls={row['prefill_calls']} avg_batch={row['prefill_batch_avg']:.2f} "
        f"time={row['prefill_s']:.2f}s tok/s={row['prefill_tok_s']:.1f} | "
        f"decode steps={row['decode_steps']} time={row['decode_s']:.2f}s "
        f"step={row['decode_step_ms']:.1f}ms slot-step={row['decode_slot_step_ms']:.1f}ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--device_ids", default="0", type=str)
    parser.add_argument("--max_slots", default=8, type=int)
    parser.add_argument("--arrival_rates", default="0.5,1,2", type=str)
    parser.add_argument("--duration", default=30.0, type=float)
    parser.add_argument("--prompt_lens", default="128", type=str)
    parser.add_argument("--output_lens", default="32", type=str)
    parser.add_argument("--workload", default="", type=str)
    parser.add_argument("--num_requests", default=0, type=int)
    parser.add_argument("--arrival_pattern", default="fixed", choices=["fixed", "poisson"])
    parser.add_argument(
        "--prefill_wait_ms",
        default=0.0,
        type=float,
        help="Optional scheduler coalescing delay before admitting waiting requests to improve batch prefill.",
    )
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--csv", default="bench_online_cb.csv", type=str)
    eos_group = parser.add_mutually_exclusive_group()
    eos_group.add_argument("--ignore_eos", dest="ignore_eos", action="store_true", default=True)
    eos_group.add_argument("--respect_eos", dest="ignore_eos", action="store_false")
    args = parser.parse_args()

    device_ids = parse_int_list(args.device_ids)
    if len(device_ids) != 1:
        print("Warning: continuous batching demo is validated for single GPU; running with provided device_ids.")

    model = Qwen2TP(Path(args.model), llaisys.DeviceType.NVIDIA, device_ids=device_ids)
    rows = []
    for arrival_rate in parse_float_list(args.arrival_rates):
        print(f"\n=== Online Continuous Batching: arrival_rate={arrival_rate} req/s ===")
        row = run_one(model, args, arrival_rate, device_ids)
        rows.append(row)
        print_summary(row)

    fields = [
        "arrival_rate",
        "qps",
        "completed",
        "output_tok_s",
        "total_tok_s",
        "avg_ttft_ms",
        "p50_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
        "avg_tpot_ms",
        "p99_latency_ms",
        "peak_mem_mib",
        "avg_active",
        "max_active",
        "wall_s",
        "prompt_len_avg",
        "prompt_len_p90",
        "output_len_avg",
        "prefill_calls",
        "prefill_batch_avg",
        "prefill_s",
        "prefill_tok_s",
        "decode_steps",
        "decode_s",
        "decode_step_ms",
        "decode_slot_step_ms",
    ]
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote summary CSV to {args.csv}")


if __name__ == "__main__":
    main()
