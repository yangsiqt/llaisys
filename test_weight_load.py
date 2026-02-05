#!/usr/bin/env python
"""Test weight loading step by step"""
import llaisys
import torch
import safetensors
from pathlib import Path
import numpy as np

print("Step 1: Load one weight from safetensors...")
model_path = Path("models/DeepSeek-R1-Distill-Qwen-1.5B")
file = list(model_path.glob("*.safetensors"))[0]
print(f"  File: {file}")

data = safetensors.safe_open(str(file), framework="pt", device="cpu")
keys = list(data.keys())
print(f"  Found {len(keys)} weights")

# Test loading first weight
test_key = keys[0]
print(f"\nStep 2: Loading weight '{test_key}'...")
weight_torch = data.get_tensor(test_key)
print(f"  Shape: {weight_torch.shape}, dtype: {weight_torch.dtype}")

# Convert to numpy
if weight_torch.dtype == torch.bfloat16:
    weight_np = weight_torch.view(torch.uint16).numpy()
    print(f"  Converted bf16 to uint16, shape: {weight_np.shape}")
elif weight_torch.dtype == torch.float16:
    weight_np = weight_torch.view(torch.int16).numpy().view(np.uint16)
    print(f"  Converted fp16 to uint16, shape: {weight_np.shape}")
else:
    weight_np = weight_torch.numpy()
    print(f"  Numpy dtype: {weight_np.dtype}")

weight_np = np.ascontiguousarray(weight_np)
print(f"  Is contiguous: {weight_np.flags['C_CONTIGUOUS']}")
print(f"  Data pointer: {weight_np.ctypes.data}")

print("\nStep 3: Create LLAISYS tensor...")
tensor = llaisys.Tensor(tuple(weight_np.shape), dtype=5, device=llaisys.DeviceType.CPU)
print(f"  Tensor created, shape: {tensor.shape()}")

print("\nStep 4: Load data into tensor...")
tensor.load(weight_np.ctypes.data)
print("  ✓ Data loaded successfully!")

print("\nStep 5: Verify data...")
data_back = np.array(tensor.to_host(), dtype=np.uint16).reshape(weight_np.shape)
matches = np.allclose(weight_np[:10].flatten(), data_back[:10].flatten())
print(f"  First 10 elements match: {matches}")
if matches:
    print("  ✓ Weight loading works!")
else:
    print(f"  Original: {weight_np[:10].flatten()}")
    print(f"  Loaded:   {data_back[:10].flatten()}")


