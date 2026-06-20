#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_DIR="${PROFILE_DIR:-${ROOT_DIR}/profiles}"
MODEL_PATH="${MODEL_PATH:-/model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
REPORT_NAME="${REPORT_NAME:-qwen2_p128_d128}"

mkdir -p "${PROFILE_DIR}"
cd "${ROOT_DIR}"

nsys profile \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=true \
  --output="${PROFILE_DIR}/${REPORT_NAME}" \
  python3 test/profile_qwen2.py \
    --model "${MODEL_PATH}" \
    --prompt-length 128 \
    --output-length 128 \
    --capture \
    --output "${PROFILE_DIR}/${REPORT_NAME}_metadata.json"

nsys stats \
  --force-export=true \
  --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,nvtx_gpu_proj_sum,nvtx_kern_sum \
  --format csv \
  --output "${PROFILE_DIR}/${REPORT_NAME}_stats" \
  "${PROFILE_DIR}/${REPORT_NAME}.nsys-rep"

python3 scripts/summarize_nsys.py \
  --stats-prefix "${PROFILE_DIR}/${REPORT_NAME}_stats" \
  --metadata "${PROFILE_DIR}/${REPORT_NAME}_metadata.json" \
  --output "${PROFILE_DIR}/${REPORT_NAME}_analysis.md" \
  --top 10
