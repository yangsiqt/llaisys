#!/usr/bin/env python
"""Debug Qwen2.__init__() step by step"""
import sys
import gc
from pathlib import Path
import json
import torch
import safetensors.torch
import numpy as np
import llaisys
from llaisys.libllaisys.qwen2 import LlaisysQwen2Meta
from llaisys.tensor import Tensor
from ctypes import byref, c_int

print("="*60)
print("SIMULATING Qwen2.__init__()")
print("="*60)

model_path = Path("models/DeepSeek-R1-Distill-Qwen-1.5B")

# Step 1: Load config
print("\nStep 1: Load config...")
with open(model_path / "config.json", "r") as f:
    config = json.load(f)
print("✓")

# Step 2: Extract metadata
print("\nStep 2: Extract metadata...")
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
dtype = 5  # BF16
print(f"✓ nlayer={nlayer}, voc={voc}")

# Step 3: Create meta
print("\nStep 3: Create C metadata...")
meta = LlaisysQwen2Meta()
meta.dtype = dtype
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
print("✓")

# Step 4: Create model
print("\nStep 4: Create model...")
device_id = 0
device_ids = (c_int * 1)(device_id)
model = llaisys.libllaisys.LIB_LLAISYS.llaisysQwen2ModelCreate(
    byref(meta), 0, device_ids, 1
)
print(f"✓ model={model}")

# Step 5: Get weights
print("\nStep 5: Get weights structure...")
weights_ptr = llaisys.libllaisys.LIB_LLAISYS.llaisysQwen2ModelWeights(model)
weights = weights_ptr.contents
print("✓")

# Step 6: Load weights (SIMPLIFIED - only first 3)
print("\nStep 6: Load weights (first 3 only)...")
weight_count = 0
max_weights = 3

for file in sorted(model_path.glob("*.safetensors")):
    print(f"  Loading from {file.name}...")
    weight_dict = safetensors.torch.load_file(str(file), device='cpu')
    
    for name, weight_torch in weight_dict.items():
        if weight_count >= max_weights:
            break
            
        print(f"    [{weight_count+1}] {name}: {weight_torch.shape}")
        
        # Convert
        if weight_torch.dtype == torch.bfloat16:
            weight_np = weight_torch.view(torch.uint16).numpy()
        elif weight_torch.dtype == torch.float16:
            weight_np = weight_torch.view(torch.int16).numpy().view(np.uint16)
        else:
            weight_np = weight_torch.numpy()
        
        weight_count += 1
        weight_np = np.ascontiguousarray(weight_np)
        
        # Create Tensor - THIS IS THE KEY DIFFERENCE
        # In Qwen2.__init__(), it creates Tensor with self._dtype, self._device
        print(f"      Creating LLAISYS tensor...")
        tensor = Tensor(tuple(weight_np.shape), dtype=dtype, device=llaisys.DeviceType.CPU)
        
        print(f"      Loading data...")
        tensor.load(weight_np.ctypes.data)
        
        # Map to weights structure
        if name == "model.embed_tokens.weight":
            print(f"      Assigning to weights.in_embed...")
            weights.in_embed = tensor._tensor  # THIS LINE!
            print(f"      ✓ Assigned")
        
        del weight_torch, weight_np
        gc.collect()
    
    if weight_count >= max_weights:
        break

print(f"\n✓ Loaded {weight_count} weights")

# Step 7: Cleanup
print("\nStep 7: Cleanup...")
llaisys.libllaisys.LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
print("✓")

print("\n" + "="*60)
print("✅ SIMULATION COMPLETE!")
print("="*60)


