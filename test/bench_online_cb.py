import argparse
import csv
import json
import random
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

import llaisys
from llaisys.models import Qwen2TP, Qwen2TPContinuousEngine

DEFAULT_RESULT_DIR = Path("/home/dzy/za/tmp")


def parse_int_list(value):
    return [int(x) for x in value.split(",") if x.strip()]


def parse_float_list(value):
    values = []
    for x in value.split(","):
        x = x.strip()
        if not x:
            continue
        values.append(float("inf") if x.lower() == "inf" else float(x))
    return values


def format_rate(value):
    return "inf" if value == float("inf") else f"{value:.2f}"


def sample_len(rng, choices, range_value):
    if range_value:
        vals = parse_int_list(range_value)
        if len(vals) != 2:
            raise ValueError("length range must be formatted as low,high")
        lo, hi = vals
        return rng.randint(lo, hi)
    return rng.choice(choices)


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


def build_requests(arrival_rate, duration_s, num_requests, prompt_lens, output_lens,
                   prompt_len_range, output_len_range, vocab_size, eos_token_id, rng):
    requests = []
    t = 0.0
    req_id = 0
    interval = 0.0 if arrival_rate == float("inf") else 1.0 / arrival_rate
    while (num_requests > 0 and req_id < num_requests) or (num_requests <= 0 and t < duration_s):
        prompt_len = sample_len(rng, prompt_lens, prompt_len_range)
        output_len = sample_len(rng, output_lens, output_len_range)
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
    interval = 0.0 if arrival_rate == float("inf") else 1.0 / arrival_rate
    for i, rec in enumerate(records):
        if arrival_rate != float("inf") and t >= duration_s:
            break
        req = {
            "request_id": int(rec.get("request_id", i)),
            "arrival_s": t,
            "prompt_tokens": rec["prompt_tokens"],
            "prompt_len": rec["prompt_len"],
            "output_len": rec["output_len"],
        }
        requests.append(req)
        if arrival_rate == float("inf"):
            t = 0.0
        elif pattern == "poisson":
            t += rng.expovariate(arrival_rate)
        else:
            t += interval
    return requests


def choose_prefill_len(waiting, prefill_bucket_size):
    if not waiting or prefill_bucket_size <= 0:
        return waiting[0]["prompt_len"]
    first_len = waiting[0]["prompt_len"]
    bucket_start = (first_len // prefill_bucket_size) * prefill_bucket_size
    bucket_end = bucket_start + prefill_bucket_size
    counts = Counter(
        req["prompt_len"]
        for req in waiting
        if bucket_start <= req["prompt_len"] < bucket_end
    )
    if not counts:
        return first_len
    return max(counts.items(), key=lambda item: (item[1], -abs(item[0] - first_len)))[0]


def running_count(engine):
    return engine.active_count() + engine.prefilling_count()


def running_room(engine, max_running):
    if max_running <= 0:
        return len(engine.free_slots)
    return max(0, min(len(engine.free_slots), max_running - running_count(engine)))


def pick_paged_varlen_batch(waiting, free_room, max_prefill_batch, scratch_slots,
                            bucket_size, max_padded_tokens):
    if not waiting or free_room <= 0:
        return [], waiting, 0
    limit = min(free_room, max_prefill_batch, scratch_slots)
    if limit <= 0:
        return [], waiting, 0

    if bucket_size > 0:
        buckets = Counter(req["prompt_len"] // bucket_size for req in waiting)
        bucket_id = max(buckets.items(), key=lambda item: (item[1], -item[0]))[0]
        candidates = [req for req in waiting if req["prompt_len"] // bucket_size == bucket_id]
    else:
        candidates = list(waiting)
    candidates = sorted(candidates, key=lambda req: req["prompt_len"])

    best = []
    best_score = None
    for start in range(len(candidates)):
        batch = []
        max_len = 0
        for req in candidates[start:]:
            if len(batch) >= limit:
                break
            max_len = max(max_len, req["prompt_len"])
            padded_tokens = max_len * (len(batch) + 1)
            if max_padded_tokens > 0 and padded_tokens > max_padded_tokens and batch:
                break
            batch.append(req)
        if not batch:
            continue
        padded_tokens = max(req["prompt_len"] for req in batch) * len(batch)
        real_tokens = sum(req["prompt_len"] for req in batch)
        score = (len(batch), real_tokens / max(1, padded_tokens), -padded_tokens)
        if best_score is None or score > best_score:
            best_score = score
            best = batch

    if not best:
        best = [min(waiting, key=lambda req: req["prompt_len"])]
    chosen_ids = {id(req) for req in best}
    keep = [req for req in waiting if id(req) not in chosen_ids]
    padded_tokens = max(req["prompt_len"] for req in best) * len(best)
    return best, keep, padded_tokens


def run_one(model, args, arrival_rate, device_ids):
    rate_seed = 1000000 if arrival_rate == float("inf") else int(arrival_rate * 1000)
    rng = random.Random(args.seed + rate_seed)
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
            args.num_requests,
            parse_int_list(args.prompt_lens),
            parse_int_list(args.output_lens),
            args.prompt_len_range,
            args.output_len_range,
            vocab_size,
            eos_token_id,
            rng,
        )

    engine = Qwen2TPContinuousEngine(
        model,
        max_slots=args.max_slots,
        eos_token_id=eos_token_id,
        ignore_eos=args.ignore_eos,
        kv_mode=args.kv_mode,
        paged_block_size=args.paged_block_size,
        paged_max_blocks=args.paged_max_blocks,
        paged_prefill_scratch_slots=args.paged_prefill_scratch_slots,
    )
    sampler = GpuMemPeakSampler(device_ids)
    waiting = []
    completed = []
    active_samples = []
    waiting_samples = []
    req_idx = 0
    last_progress_s = 0.0

    start = time.perf_counter()
    sampler.start()
    try:
        while req_idx < len(requests) or waiting or engine.prefilling_count() > 0 or engine.active_count() > 0:
            now = time.perf_counter()
            elapsed = now - start

            while req_idx < len(requests) and requests[req_idx]["arrival_s"] <= elapsed:
                req = requests[req_idx]
                req["arrival_abs_s"] = start + req["arrival_s"]
                waiting.append(req)
                req_idx += 1

            prefill_budget = args.chunked_prefill_tokens
            if prefill_budget > 0 and engine.prefilling_count() > 0:
                done, used_tokens = engine.prefill_step(prefill_budget)
                completed.extend(done)
                prefill_budget -= used_tokens

            while waiting and engine.can_accept() and running_room(engine, args.max_running) > 0:
                if args.chunked_prefill_tokens > 0:
                    if args.kv_mode == "paged" and engine.prefilling_count() > 0:
                        break
                    if args.max_running > 0 and running_count(engine) >= args.max_running:
                        break
                    if prefill_budget <= 0:
                        break
                    req = waiting.pop(0)
                    done, used_tokens = engine.start_prefill_chunked(
                        req, prefill_budget, req["arrival_abs_s"]
                    )
                    completed.extend(done)
                    prefill_budget -= used_tokens
                    continue
                if args.prefill_wait_ms > 0 and req_idx < len(requests):
                    oldest_wait_s = time.perf_counter() - waiting[0]["arrival_abs_s"]
                    free_slots = len(engine.free_slots)
                    if oldest_wait_s * 1000.0 < args.prefill_wait_ms and len(waiting) < free_slots:
                        break
                free_slots = running_room(engine, args.max_running)
                if free_slots <= 0:
                    break
                max_prefill_batch = args.max_prefill_batch if args.max_prefill_batch > 0 else free_slots
                if args.kv_mode == "paged":
                    batch, waiting, used_prefill_tokens = pick_paged_varlen_batch(
                        waiting,
                        free_slots,
                        max_prefill_batch,
                        args.paged_prefill_scratch_slots,
                        args.prefill_bucket_size,
                        args.max_prefill_padded_tokens,
                    )
                    if not batch:
                        break
                    completed.extend(engine.submit_many_varlen(batch, time.perf_counter()))
                    continue

                first_len = choose_prefill_len(waiting, args.prefill_bucket_size)
                batch = []
                keep = []
                used_prefill_tokens = 0
                for req in waiting:
                    same_len = req["prompt_len"] == first_len
                    under_batch = len(batch) < min(free_slots, max_prefill_batch)
                    under_budget = (
                        prefill_budget <= 0
                        or used_prefill_tokens + req["prompt_len"] <= prefill_budget
                        or not batch
                    )
                    if same_len and under_batch and under_budget:
                        batch.append(req)
                        used_prefill_tokens += req["prompt_len"]
                    else:
                        keep.append(req)
                waiting = keep
                if not batch:
                    break
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
                if prefill_budget > 0:
                    prefill_budget -= used_prefill_tokens
                    if prefill_budget <= 0:
                        break

            if engine.active_count() > 0:
                decode_steps_this_loop = max(1, args.decode_steps_per_loop)
                for _ in range(decode_steps_this_loop):
                    if engine.active_count() <= 0:
                        break
                    completed.extend(engine.step(time.perf_counter(), max_batch=args.max_active_decode))
                    active_samples.append(engine.active_count())
                    waiting_samples.append(len(waiting))
            elif req_idx < len(requests):
                sleep_s = max(0.0, start + requests[req_idx]["arrival_s"] - time.perf_counter())
                time.sleep(min(sleep_s, 0.001))

            if args.progress_interval > 0 and elapsed - last_progress_s >= args.progress_interval:
                last_progress_s = elapsed
                cur_output = sum(len(c["output_tokens"]) for c in completed)
                cur_wall = max(1e-9, time.perf_counter() - start)
                print(
                    f"[progress] t={elapsed:.1f}s submitted={req_idx}/{len(requests)} "
                    f"waiting={len(waiting)} prefilling={engine.prefilling_count()} active={engine.active_count()} "
                    f"completed={len(completed)} output_tok_s={cur_output / cur_wall:.1f}",
                    flush=True,
                )
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
    decode_batch_avg = (
        engine_metrics["decode_slot_steps"] / engine_metrics["decode_steps"]
        if engine_metrics["decode_steps"] > 0
        else 0.0
    )

    row = {
        "arrival_rate": arrival_rate,
        "kv_mode": args.kv_mode,
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
        "avg_waiting": sum(waiting_samples) / len(waiting_samples) if waiting_samples else 0.0,
        "max_waiting": max(waiting_samples) if waiting_samples else 0,
        "wall_s": wall_s,
        "prompt_len_avg": sum(prompt_lens_completed) / len(prompt_lens_completed) if prompt_lens_completed else 0.0,
        "prompt_len_p90": percentile(prompt_lens_completed, 90),
        "output_len_avg": sum(output_lens_completed) / len(output_lens_completed) if output_lens_completed else 0.0,
        "prefill_calls": engine_metrics["prefill_calls"],
        "prefill_chunks": engine_metrics.get("prefill_chunks", 0),
        "chunked_prefill_reqs": engine_metrics.get("chunked_prefill_reqs", 0),
        "prefill_batch_avg": prefill_batch_avg,
        "prefill_s": prefill_s,
        "prefill_tok_s": prefill_tok_s,
        "decode_steps": engine_metrics["decode_steps"],
        "decode_batch_avg": decode_batch_avg,
        "decode_max_batch": engine_metrics["decode_max_batch"],
        "decode_s": decode_s,
        "decode_step_ms": decode_step_s * 1000.0,
        "decode_slot_step_ms": decode_slot_step_s * 1000.0,
        "max_active_decode": args.max_active_decode,
        "max_running": args.max_running,
        "decode_steps_per_loop": args.decode_steps_per_loop,
        "max_prefill_batch": args.max_prefill_batch,
        "chunked_prefill_tokens": args.chunked_prefill_tokens,
        "prefill_bucket_size": args.prefill_bucket_size,
        "max_prefill_padded_tokens": args.max_prefill_padded_tokens,
    }
    row.update(
        {
            "block_size": engine_metrics.get("paged_block_size", 0),
            "max_blocks": engine_metrics.get("paged_max_blocks", 0),
            "used_blocks": engine_metrics.get("paged_used_blocks", 0),
            "peak_used_blocks": engine_metrics.get("paged_peak_used_blocks", 0),
            "free_blocks": engine_metrics.get("paged_free_blocks", 0),
            "kv_capacity_tokens": engine_metrics.get("paged_kv_capacity_tokens", 0),
            "block_utilization": engine_metrics.get("paged_block_utilization", 0.0),
        }
    )
    return row


def print_summary(row):
    print(
        f"arrival={format_rate(row['arrival_rate'])} req/s | completed={row['completed']} | "
        f"qps={row['qps']:.2f} | output={row['output_tok_s']:.2f} tok/s | "
        f"TTFT avg/p90/p99={row['avg_ttft_ms']:.1f}/{row['p90_ttft_ms']:.1f}/{row['p99_ttft_ms']:.1f} ms | "
        f"TPOT={row['avg_tpot_ms']:.1f} ms | p99 latency={row['p99_latency_ms']:.1f} ms | "
        f"active avg/max={row['avg_active']:.1f}/{row['max_active']} | "
        f"waiting avg/max={row['avg_waiting']:.1f}/{row['max_waiting']} | mem={row['peak_mem_mib']} MiB"
    )
    if row["kv_mode"] == "paged":
        print(
            f"  paged kv: block_size={row['block_size']} max_blocks={row['max_blocks']} "
            f"used/peak/free={row['used_blocks']}/{row['peak_used_blocks']}/{row['free_blocks']} "
            f"capacity_tokens={row['kv_capacity_tokens']} utilization={row['block_utilization']:.3f}"
        )
    print(
        f"  profile: wall={row['wall_s']:.2f}s | prompt avg/p90={row['prompt_len_avg']:.1f}/{row['prompt_len_p90']:.0f} | "
        f"prefill calls={row['prefill_calls']} avg_batch={row['prefill_batch_avg']:.2f} "
        f"chunks={row['prefill_chunks']} chunked_reqs={row['chunked_prefill_reqs']} "
        f"time={row['prefill_s']:.2f}s tok/s={row['prefill_tok_s']:.1f} | "
        f"decode steps={row['decode_steps']} batch avg/max={row['decode_batch_avg']:.1f}/{row['decode_max_batch']} "
        f"time={row['decode_s']:.2f}s "
        f"step={row['decode_step_ms']:.1f}ms slot-step={row['decode_slot_step_ms']:.1f}ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--device_ids", default="0", type=str)
    parser.add_argument("--max_slots", default=8, type=int)
    parser.add_argument("--kv_mode", default="fixed", choices=["fixed", "paged"])
    parser.add_argument("--paged_block_size", default=16, type=int)
    parser.add_argument("--paged_max_blocks", default=0, type=int)
    parser.add_argument("--paged_prefill_scratch_slots", default=1, type=int)
    parser.add_argument("--arrival_rates", default="0.5,1,2", type=str)
    parser.add_argument("--duration", default=30.0, type=float)
    parser.add_argument("--prompt_lens", default="128", type=str)
    parser.add_argument("--output_lens", default="32", type=str)
    parser.add_argument("--prompt_len_range", default="", type=str)
    parser.add_argument("--output_len_range", default="", type=str)
    parser.add_argument("--workload", default="", type=str)
    parser.add_argument("--num_requests", default=0, type=int)
    parser.add_argument("--arrival_pattern", default="fixed", choices=["fixed", "poisson"])
    parser.add_argument("--max_active_decode", default=0, type=int)
    parser.add_argument("--max_running", default=0, type=int)
    parser.add_argument("--decode_steps_per_loop", default=1, type=int)
    parser.add_argument("--prefill_bucket_size", default=0, type=int)
    parser.add_argument("--max_prefill_batch", default=0, type=int)
    parser.add_argument("--max_prefill_padded_tokens", default=0, type=int)
    parser.add_argument("--chunked_prefill_tokens", default=0, type=int)
    parser.add_argument("--progress_interval", default=10.0, type=float)
    parser.add_argument(
        "--prefill_wait_ms",
        default=0.0,
        type=float,
        help="Optional scheduler coalescing delay before admitting waiting requests to improve batch prefill.",
    )
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--csv", default="", type=str)
    eos_group = parser.add_mutually_exclusive_group()
    eos_group.add_argument("--ignore_eos", dest="ignore_eos", action="store_true", default=True)
    eos_group.add_argument("--respect_eos", dest="ignore_eos", action="store_false")
    args = parser.parse_args()
    if args.kv_mode == "paged":
        if len(parse_int_list(args.device_ids)) != 1:
            raise ValueError("kv_mode=paged is TP=1 only in the MVP")
        if args.paged_max_blocks <= 0:
            raise ValueError("--paged_max_blocks must be > 0 when --kv_mode paged")
    if not args.csv:
        DEFAULT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        args.csv = str(DEFAULT_RESULT_DIR / "bench_online_cb.csv")
    else:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            DEFAULT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
            args.csv = str(DEFAULT_RESULT_DIR / csv_path)
        else:
            csv_path.parent.mkdir(parents=True, exist_ok=True)

    device_ids = parse_int_list(args.device_ids)
    if len(device_ids) != 1:
        print("Warning: continuous batching demo is validated for single GPU; running with provided device_ids.")

    model = Qwen2TP(Path(args.model), llaisys.DeviceType.NVIDIA, device_ids=device_ids)
    rows = []
    for arrival_rate in parse_float_list(args.arrival_rates):
        print(f"\n=== Online Continuous Batching: arrival_rate={format_rate(arrival_rate)} req/s ===")
        row = run_one(model, args, arrival_rate, device_ids)
        rows.append(row)
        print_summary(row)

    fields = [
        "arrival_rate",
        "kv_mode",
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
        "avg_waiting",
        "max_waiting",
        "wall_s",
        "prompt_len_avg",
        "prompt_len_p90",
        "output_len_avg",
        "prefill_calls",
        "prefill_chunks",
        "chunked_prefill_reqs",
        "prefill_batch_avg",
        "prefill_s",
        "prefill_tok_s",
        "decode_steps",
        "decode_batch_avg",
        "decode_max_batch",
        "decode_s",
        "decode_step_ms",
        "decode_slot_step_ms",
        "max_active_decode",
        "max_running",
        "decode_steps_per_loop",
        "max_prefill_batch",
        "chunked_prefill_tokens",
        "prefill_bucket_size",
        "max_prefill_padded_tokens",
        "block_size",
        "max_blocks",
        "used_blocks",
        "peak_used_blocks",
        "free_blocks",
        "kv_capacity_tokens",
        "block_utilization",
    ]
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote summary CSV to {args.csv}")


if __name__ == "__main__":
    main()
