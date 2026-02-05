#!/usr/bin/env python
"""Test Qwen2 initialization in detail"""
import sys
import json
from pathlib import Path
from ctypes import byref, c_int
import llaisys
from llaisys.libllaisys.qwen2 import LlaisysQwen2Meta

print("Step 1: Load config...")
model_path = Path("models/DeepSeek-R1-Distill-Qwen-1.5B")
with open(model_path / "config.json", "r") as f:
    config = json.load(f)
print("  ✓ Config loaded")

print("\nStep 2: Prepare metadata...")
nlayer = config["num_hidden_layers"]
hs = config["hidden_size"]
nh = config["num_attention_heads"]
nkvh = config["num_key_value_heads"]
dh = hs // nh
di = config["intermediate_size"]
maxseq = min(config.get("max_position_embeddings", 131072), 4096)
voc = config["vocab_size"]
epsilon = config["rms_norm_eps"]
theta = config.get("rope_theta", 10000.0)
end_token = config["eos_token_id"]

print(f"  nlayer={nlayer}, hs={hs}, voc={voc}")

print("\nStep 3: Create C metadata structure...")
meta = LlaisysQwen2Meta()
meta.dtype = 5  # BF16
meta.nlayer = nlayer
meta.hs = hs
meta.nh = nh
meta.nkvh = nkvh
meta.dh = dh
meta.di = di
meta.maxseq = maxseq
meta.voc = voc
meta.epsilon = epsilon
meta.theta = theta
meta.end_token = end_token
print("  ✓ Metadata created")

print("\nStep 4: Create model...")
device_ids = (c_int * 1)(0)
model = llaisys.libllaisys.LIB_LLAISYS.llaisysQwen2ModelCreate(
    byref(meta), 0, device_ids, 1
)
print(f"  ✓ Model handle: {model}")

print("\nStep 5: Get weights structure...")
weights_ptr = llaisys.libllaisys.LIB_LLAISYS.llaisysQwen2ModelWeights(model)
print(f"  ✓ Weights pointer: {weights_ptr}")

if weights_ptr:
    weights = weights_ptr.contents
    print("\nStep 6: Check weights structure...")
    print(f"  in_embed: {weights.in_embed}")
    print(f"  out_embed: {weights.out_embed}")
    print(f"  out_norm_w: {weights.out_norm_w}")
    print(f"  attn_norm_w: {weights.attn_norm_w}")
    print("  ✓ Weights structure accessible")

print("\nStep 7: Cleanup...")
llaisys.libllaisys.LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
print("  ✓ Model destroyed")

print("\n" + "="*60)
print("✓ Model initialization test passed!")
print("="*60)


