#!/usr/bin/env python
"""Step by step debugging"""
import sys
import gc

print("Step 1: Import llaisys...")
import llaisys
print("  ✓ OK")

print("\nStep 2: Import torch...")
import torch
print("  ✓ OK")

print("\nStep 3: Import safetensors...")
import safetensors.torch
print("  ✓ OK")

print("\nStep 4: Load one weight file...")
try:
    from pathlib import Path
    model_path = Path("models/DeepSeek-R1-Distill-Qwen-1.5B")
    file = list(model_path.glob("*.safetensors"))[0]
    print(f"  File: {file}")
    
    print("\nStep 5: Load with safetensors.torch...")
    weight_dict = safetensors.torch.load_file(str(file), device='cpu')
    print(f"  ✓ Loaded {len(weight_dict)} tensors")
    
    # Get first weight
    first_key = list(weight_dict.keys())[0]
    first_weight = weight_dict[first_key]
    print(f"\nStep 6: First weight info:")
    print(f"  Key: {first_key}")
    print(f"  Shape: {first_weight.shape}")
    print(f"  Dtype: {first_weight.dtype}")
    
    # Convert to numpy
    print("\nStep 7: Convert to numpy...")
    if first_weight.dtype == torch.bfloat16:
        weight_np = first_weight.view(torch.uint16).numpy()
    else:
        weight_np = first_weight.numpy()
    print(f"  ✓ Numpy shape: {weight_np.shape}, dtype: {weight_np.dtype}")
    
    # Clear memory
    del weight_dict, first_weight, weight_np
    gc.collect()
    print("\nStep 8: Cleared memory")
    
    print("\nStep 9: Create LLAISYS model (without loading weights)...")
    # Just test model creation
    from llaisys.libllaisys.qwen2 import LlaisysQwen2Meta
    from ctypes import byref, c_int
    
    meta = LlaisysQwen2Meta()
    meta.dtype = 5  # BF16
    meta.nlayer = 28
    meta.hs = 1536
    meta.nh = 12
    meta.nkvh = 2
    meta.dh = 128
    meta.di = 8960
    meta.maxseq = 4096
    meta.voc = 151936
    meta.epsilon = 1e-6
    meta.theta = 10000.0
    meta.end_token = 151643
    
    device_ids = (c_int * 1)(0)
    model = llaisys.libllaisys.LIB_LLAISYS.llaisysQwen2ModelCreate(
        byref(meta), 0, device_ids, 1
    )
    print(f"  ✓ Model created: {model}")
    
    if model:
        llaisys.libllaisys.LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
        print("  ✓ Model destroyed")
    
    print("\n" + "="*60)
    print("✓ All steps completed successfully!")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


