import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def parse_int_list(value):
    if not value:
        return []
    return [int(x) for x in value.split(",") if x.strip()]


def load_json_or_jsonl(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ("data", "train", "items", "samples"):
            if isinstance(data.get(key), list):
                return data[key]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported dataset root type in {path}")
    return data


def normalize_role(role):
    role = str(role or "").lower()
    if role in ("human", "user", "prompter"):
        return "user"
    if role in ("gpt", "assistant", "bot"):
        return "assistant"
    return role


def extract_prompt_answer(item):
    conversations = item.get("conversations") or item.get("messages")
    if isinstance(conversations, list) and len(conversations) >= 2:
        user_text = None
        assistant_text = None
        for msg in conversations:
            role = normalize_role(msg.get("from", msg.get("role")))
            value = msg.get("value", msg.get("content", ""))
            if role == "user" and user_text is None:
                user_text = value
            elif role == "assistant" and user_text is not None:
                assistant_text = value
                break
        if user_text and assistant_text:
            return str(user_text), str(assistant_text), "conversation"

    if "instruction" in item and "output" in item:
        prompt = str(item.get("instruction", ""))
        if item.get("input"):
            prompt = prompt + "\n" + str(item["input"])
        return prompt, str(item.get("output", "")), "alpaca"

    if "prompt" in item and ("completion" in item or "response" in item or "answer" in item):
        answer = item.get("completion", item.get("response", item.get("answer", "")))
        return str(item.get("prompt", "")), str(answer), "prompt_completion"

    return None


def build_records(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)
    raw_items = load_json_or_jsonl(args.dataset)
    rng.shuffle(raw_items)
    target_prompt_lens = parse_int_list(args.target_prompt_lens)

    records = []
    for item in raw_items:
        pair = extract_prompt_answer(item)
        if pair is None:
            continue
        prompt, answer, source_format = pair
        prompt = prompt.strip()
        answer = answer.strip()
        if not prompt or not answer:
            continue

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        if target_prompt_lens:
            eligible_lens = [x for x in target_prompt_lens if len(prompt_ids) >= x]
            if not eligible_lens:
                continue
            target_prompt_len = rng.choice(eligible_lens)
            prompt_ids = prompt_ids[:target_prompt_len]
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        else:
            if not (args.min_prompt_len <= len(prompt_ids) <= args.max_prompt_len):
                continue
        if len(answer_ids) < args.min_output_len:
            continue

        output_len = min(len(answer_ids), args.max_output_len)
        records.append(
            {
                "request_id": len(records),
                "prompt": prompt,
                "reference_output": answer,
                "prompt_tokens": [int(x) for x in prompt_ids],
                "prompt_len": len(prompt_ids),
                "output_len": int(output_len),
                "reference_output_len": len(answer_ids),
                "source_format": source_format,
            }
        )
        if len(records) >= args.num_requests:
            break

    if not records:
        raise RuntimeError("No usable requests after filtering; relax length limits or check dataset format")
    return records


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_vllm_sharegpt(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in records:
        rows.append(
            {
                "id": str(rec["request_id"]),
                "conversations": [
                    {"from": "human", "value": rec["prompt"]},
                    {"from": "gpt", "value": rec["reference_output"]},
                ],
            }
        )
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", default="test/data/real_workload_small.jsonl")
    parser.add_argument("--vllm_sharegpt_out", default="")
    parser.add_argument("--num_requests", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min_prompt_len", type=int, default=16)
    parser.add_argument("--max_prompt_len", type=int, default=2048)
    parser.add_argument(
        "--target_prompt_lens",
        default="",
        help="Comma-separated exact prompt lengths. Prompts longer than a chosen target are truncated and decoded back to text.",
    )
    parser.add_argument("--min_output_len", type=int, default=16)
    parser.add_argument("--max_output_len", type=int, default=256)
    args = parser.parse_args()

    records = build_records(args)
    write_jsonl(args.out, records)
    if args.vllm_sharegpt_out:
        write_vllm_sharegpt(args.vllm_sharegpt_out, records)

    prompt_lens = [r["prompt_len"] for r in records]
    output_lens = [r["output_len"] for r in records]
    print(f"Wrote {len(records)} requests to {args.out}")
    print(f"prompt_len min/avg/max = {min(prompt_lens)}/{sum(prompt_lens)/len(prompt_lens):.1f}/{max(prompt_lens)}")
    print(f"output_len min/avg/max = {min(output_lens)}/{sum(output_lens)/len(output_lens):.1f}/{max(output_lens)}")
    if args.vllm_sharegpt_out:
        print(f"Wrote vLLM ShareGPT-compatible data to {args.vllm_sharegpt_out}")


if __name__ == "__main__":
    main()
