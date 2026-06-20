import argparse
import csv
import json
import re
from pathlib import Path


def find_report(prefix, marker):
    parent = prefix.parent
    candidates = sorted(parent.glob(f"{prefix.name}*{marker}*.csv"))
    if not candidates:
        candidates = sorted(parent.glob(f"{prefix.name}*.csv"))
        candidates = [path for path in candidates if marker.lower() in path.name.lower()]
    if not candidates:
        raise FileNotFoundError(f"Cannot find {marker} CSV for prefix {prefix}")
    return candidates[0]


def parse_number(value):
    cleaned = re.sub(r"[^0-9.eE+-]", "", value or "")
    return float(cleaned) if cleaned else 0.0


def load_kernel_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return []
    columns = rows[0].keys()
    name_col = next(column for column in columns if "Name" in column)
    time_col = next(column for column in columns if "Total Time" in column)
    instances_col = next(
        (column for column in columns if "Instances" in column), None
    )
    percent_col = next((column for column in columns if "Time (%)" in column), None)
    result = []
    for row in rows:
        result.append(
            {
                "name": row[name_col],
                "total_ns": parse_number(row[time_col]),
                "instances": int(parse_number(row[instances_col])) if instances_col else 0,
                "percent": parse_number(row[percent_col]) if percent_col else 0.0,
            }
        )
    return sorted(result, key=lambda row: row["total_ns"], reverse=True)


def load_summary_rows(path, name_marker, time_marker, calls_marker=None):
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    result = []
    for row in rows:
        name_col = next(column for column in row if name_marker in column)
        time_col = next(column for column in row if time_marker in column)
        if calls_marker:
            calls_col = next(
                (column for column in row if calls_marker in column), None
            )
        else:
            calls_col = next(
                (
                    column
                    for column in row
                    if "Calls" in column or "Instances" in column
                ),
                None,
            )
        result.append(
            {
                "name": row[name_col],
                "total_ns": parse_number(row[time_col]),
                "calls": int(parse_number(row[calls_col])) if calls_col else 0,
            }
        )
    return sorted(result, key=lambda item: item["total_ns"], reverse=True)


def classify_kernel(name):
    lower = name.lower()
    if "gemm" in lower or "cublas" in lower:
        return "cuBLAS GEMM"
    if "self_attn" in lower or "attention" in lower:
        return "Self-attention"
    if "rms_norm" in lower:
        return "RMSNorm"
    if "swiglu" in lower:
        return "SwiGLU"
    if "rope" in lower:
        return "RoPE"
    if "argmax" in lower:
        return "Argmax"
    if "embedding" in lower:
        return "Embedding"
    if "add_bias" in lower:
        return "Bias add"
    if re.search(r"(?:^|::)add_", lower):
        return "Residual add"
    if "memcpy" in lower:
        return "Memory copy"
    return "Other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-prefix", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    prefix = Path(args.stats_prefix)
    kernel_csv = find_report(prefix, "cuda_gpu_kern_sum")
    api_csv = find_report(prefix, "cuda_api_sum")
    projection_csv = find_report(prefix, "nvtx_gpu_proj_sum")
    rows = load_kernel_rows(kernel_csv)
    api_rows = load_summary_rows(api_csv, "Name", "Total Time")
    projection_rows = load_summary_rows(
        projection_csv, "Range", "Total Proj Time", "Total GPU Ops"
    )
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))

    grouped = {}
    for row in rows:
        category = classify_kernel(row["name"])
        item = grouped.setdefault(
            category, {"total_ns": 0.0, "instances": 0, "kernels": set()}
        )
        item["total_ns"] += row["total_ns"]
        item["instances"] += row["instances"]
        item["kernels"].add(row["name"])
    categories = sorted(
        grouped.items(), key=lambda item: item[1]["total_ns"], reverse=True
    )
    total_ns = sum(item["total_ns"] for _, item in categories)
    phase_projection = {
        row["name"].lstrip(":"): row
        for row in projection_rows
        if row["name"].lstrip(":") in {"profile", "prefill", "decode"}
    }

    lines = [
        "# LLAISYS Nsys P128/D128 Analysis",
        "",
        "## Run",
        "",
        f"- Model: `{metadata['model']}`",
        f"- Device: `{metadata['device']}`",
        f"- Dtype: `{metadata['dtype']}`",
        f"- Prefill: `{metadata['prefill_ms']:.3f} ms`",
        f"- Decode: `{metadata['decode_ms']:.3f} ms`",
        f"- Mean decode step: `{metadata['decode_step_ms']:.3f} ms`",
        f"- Total CUDA kernel time: `{total_ns / 1e6:.3f} ms`",
        "",
        "## GPU-projected phase time",
        "",
        "| Phase | GPU-projected time (ms) | GPU operations |",
        "|---|---:|---:|",
    ]
    for phase in ("profile", "prefill", "decode"):
        row = phase_projection.get(phase)
        if row:
            lines.append(
                f"| {phase} | {row['total_ns'] / 1e6:.3f} | {row['calls']} |"
            )

    lines.extend(
        [
        "",
        f"## Top {args.top} kernel classes by cumulative GPU time",
        "",
        "| Rank | Kernel class | GPU time (ms) | Share | Launches | Distinct kernels |",
        "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for rank, (category, item) in enumerate(categories[: args.top], 1):
        share = item["total_ns"] / total_ns * 100.0 if total_ns else 0.0
        lines.append(
            f"| {rank} | {category} | {item['total_ns'] / 1e6:.3f} | "
            f"{share:.2f}% | {item['instances']} | {len(item['kernels'])} |"
        )

    lines.extend(
        [
            "",
            f"## Top {args.top} individual kernels",
            "",
            "| Rank | Kernel | GPU time (ms) | Share | Launches |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(rows[: args.top], 1):
        name = row["name"].replace("|", "\\|")
        lines.append(
            f"| {rank} | `{name}` | {row['total_ns'] / 1e6:.3f} | "
            f"{row['percent']:.2f}% | {row['instances']} |"
        )

    lines.extend(
        [
            "",
            "## Top CUDA API costs",
            "",
            "| Rank | API | CPU API time (ms) | Calls |",
            "|---:|---|---:|---:|",
        ]
    )
    for rank, row in enumerate(api_rows[:8], 1):
        lines.append(
            f"| {rank} | `{row['name']}` | {row['total_ns'] / 1e6:.3f} | "
            f"{row['calls']} |"
        )

    top_two_share = (
        sum(item["total_ns"] for _, item in categories[:2]) / total_ns * 100.0
        if total_ns
        else 0.0
    )
    malloc_calls = next(
        (row["calls"] for row in api_rows if row["name"] == "cudaMalloc"), 0
    )
    free_calls = next(
        (row["calls"] for row in api_rows if row["name"] == "cudaFree"), 0
    )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- GEMM and self-attention account for `{top_two_share:.2f}%` of cumulative kernel time.",
            f"- The measured region performs `{malloc_calls}` `cudaMalloc` and `{free_calls}` `cudaFree` calls; forward-workspace reuse is the clearest host-overhead target.",
            "- Decode consists of many short GEMV/GEMM and elementwise launches. Allocation reuse and kernel fusion should be evaluated before tuning the smallest kernels.",
            "- The `.nsys-rep` preserves separate NVTX ranges for prefill, decode, every decode step, every transformer layer, and the major operators.",
            "",
            "## Artifacts",
            "",
            f"- Kernel summary: `{kernel_csv.name}`",
            f"- CUDA API summary: `{api_csv.name}`",
            f"- NVTX GPU projection: `{projection_csv.name}`",
            f"- Raw metadata: `{Path(args.metadata).name}`",
            "- Open the `.nsys-rep` file in Nsight Systems for the full NVTX timeline.",
            "",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
