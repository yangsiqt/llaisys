#!/usr/bin/env python
"""Test tensor creation and loading"""
import llaisys
import numpy as np

print("Step 1: Create numpy array...")
data = np.random.randn(10, 20).astype(np.float32)
print(f"  Shape: {data.shape}, dtype: {data.dtype}")
print(f"  Is contiguous: {data.flags['C_CONTIGUOUS']}")

print("\nStep 2: Create LLAISYS tensor...")
tensor = llaisys.Tensor((10, 20), dtype=llaisys.DataType.F32, device=llaisys.DeviceType.CPU)
print(f"  Tensor created, shape: {tensor.shape()}")

print("\nStep 3: Load data...")
tensor.load(data.ctypes.data)
print("  ✓ Data loaded!")

print("\nStep 4: Read back...")
data_back = np.array(tensor.to_host(), dtype=np.float32).reshape(10, 20)
print(f"  Shape: {data_back.shape}")
print(f"  Match: {np.allclose(data, data_back)}")

if np.allclose(data, data_back):
    print("\n✓ Tensor operations work correctly!")
else:
    print("\n✗ Data mismatch!")
    print(f"  Original[0,:5]: {data[0,:5]}")
    print(f"  Loaded[0,:5]: {data_back[0,:5]}")


