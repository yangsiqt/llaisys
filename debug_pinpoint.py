#!/usr/bin/env python
"""Pinpoint the exact location of segfault"""
import sys
import gc
from pathlib import Path
import json
import torch
import safetensors.torch
import numpy as np
import llaisys
from llaisys.libllaisys.qwen2 import LlaisysQwen2Meta
from ctypes import byref, c_int

def test_phase_1():
    """Phase 1: Model creation only"""
    print("\n" + "="*60)
    print("PHASE 1: Model Creation")
    print("="*60)
    
    model_path = Path("models/DeepSeek-R1-Distill-Qwen-1.5B")
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)
    
    meta = LlaisysQwen2Meta()
    meta.dtype = 5
    meta.nlayer = config["num_hidden_layers"]
    meta.hs = config["hidden_size"]
    meta.nh = config["num_attention_heads"]
    meta.nkvh = config["num_key_value_heads"]
    meta.dh = meta.hs // meta.nh
    meta.di = config["intermediate_size"]
    meta.maxseq = 4096
    meta.voc = config["vocab_size"]
    meta.epsilon = config["rms_norm_eps"]
    meta.theta = config.get("rope_theta", 10000.0)
    meta.end_token = config["eos_token_id"]
    
    device_ids = (c_int * 1)(0)
    model = llaisys.libllaisys.LIB_LLAISYS.llaisysQwen2ModelCreate(
        byref(meta), 0, device_ids, 1
    )
    
    print(f"✓ Model created: {model}")
    return model, config

def test_phase_2(model):
    """Phase 2: Get weights structure"""
    print("\n" + "="*60)
    print("PHASE 2: Weights Structure")
    print("="*60)
    
    weights_ptr = llaisys.libllaisys.LIB_LLAISYS.llaisysQwen2ModelWeights(model)
    weights = weights_ptr.contents
    print(f"✓ Weights structure obtained")
    return weights

def test_phase_3():
    """Phase 3: Load ONE weight tensor"""
    print("\n" + "="*60)
    print("PHASE 3: Load Single Weight")
    print("="*60)
    
    model_path = Path("models/DeepSeek-R1-Distill-Qwen-1.5B")
    file = list(model_path.glob("*.safetensors"))[0]
    weight_dict = safetensors.torch.load_file(str(file), device='cpu')
    
    # Get first weight
    first_key = list(weight_dict.keys())[0]
    weight_torch = weight_dict[first_key]
    print(f"  Key: {first_key}")
    print(f"  Shape: {weight_torch.shape}")
    
    # Convert
    if weight_torch.dtype == torch.bfloat16:
        weight_np = weight_torch.view(torch.uint16).numpy()
    else:
        weight_np = weight_torch.numpy()
    weight_np = np.ascontiguousarray(weight_np)
    
    print(f"✓ Weight converted to numpy: {weight_np.shape}")
    return first_key, weight_np

def test_phase_4(weight_np):
    """Phase 4: Create LLAISYS tensor"""
    print("\n" + "="*60)
    print("PHASE 4: Create LLAISYS Tensor")
    print("="*60)
    
    print(f"  Creating tensor with shape: {weight_np.shape}")
    tensor = llaisys.Tensor(
        tuple(weight_np.shape),
        dtype=llaisys.DataType.BF16,
        device=llaisys.DeviceType.CPU
    )
    print(f"✓ Tensor created: {tensor.shape()}")
    return tensor

def test_phase_5(tensor, weight_np):
    """Phase 5: Load data into tensor"""
    print("\n" + "="*60)
    print("PHASE 5: Load Data")
    print("="*60)
    
    print(f"  Loading {weight_np.nbytes} bytes...")
    print(f"  Data pointer: {weight_np.ctypes.data}")
    
    tensor.load(weight_np.ctypes.data)
    print(f"✓ Data loaded successfully!")
    return tensor

def test_phase_6_multiple_weights():
    """Phase 6: Load MULTIPLE weights"""
    print("\n" + "="*60)
    print("PHASE 6: Load Multiple Weights")
    print("="*60)
    
    model_path = Path("models/DeepSeek-R1-Distill-Qwen-1.5B")
    file = list(model_path.glob("*.safetensors"))[0]
    weight_dict = safetensors.torch.load_file(str(file), device='cpu')
    
    tensors = []
    count = 0
    max_count = 5  # Only load 5 weights
    
    for name, weight_torch in weight_dict.items():
        if count >= max_count:
            break
        
        print(f"  [{count+1}/{max_count}] Loading {name}...")
        
        # Convert
        if weight_torch.dtype == torch.bfloat16:
            weight_np = weight_torch.view(torch.uint16).numpy()
        else:
            weight_np = weight_torch.numpy()
        weight_np = np.ascontiguousarray(weight_np)
        
        # Create tensor
        tensor = llaisys.Tensor(
            tuple(weight_np.shape),
            dtype=llaisys.DataType.BF16,
            device=llaisys.DeviceType.CPU
        )
        
        # Load data
        tensor.load(weight_np.ctypes.data)
        tensors.append(tensor)
        count += 1
        
        # Clear memory
        del weight_torch, weight_np
        gc.collect()
    
    print(f"✓ Loaded {count} weights successfully!")
    return tensors

# Run tests
try:
    print("\n" + "🔍 STARTING SYSTEMATIC DEBUG" + "\n")
    
    # Phase 1
    model, config = test_phase_1()
    
    # Phase 2
    weights = test_phase_2(model)
    
    # Phase 3
    first_key, weight_np = test_phase_3()
    
    # Phase 4
    tensor = test_phase_4(weight_np)
    
    # Phase 5
    loaded_tensor = test_phase_5(tensor, weight_np)
    
    # Cleanup before Phase 6
    del tensor, loaded_tensor, weight_np
    gc.collect()
    
    # Phase 6 - This is where it might crash
    print("\n⚠️  About to test multiple weight loading...")
    print("This is the most likely point of failure...")
    input("Press Enter to continue...")
    
    tensors = test_phase_6_multiple_weights()
    
    # Cleanup
    llaisys.libllaisys.LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
    
    print("\n" + "="*60)
    print("✅ ALL PHASES PASSED!")
    print("="*60)
    print("\nSegfault is NOT in basic operations.")
    print("It must be in the Qwen2 class initialization logic.")
    
except Exception as e:
    print(f"\n❌ ERROR in current phase: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


