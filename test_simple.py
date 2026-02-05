#!/usr/bin/env python
"""Simple test to isolate the problem"""
import llaisys

print("Step 1: Import OK")

try:
    print("Step 2: Creating config...")
    from llaisys.libllaisys.qwen2 import LlaisysQwen2Meta
    from ctypes import byref
    
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
    
    print("Step 3: Config created")
    print(f"  nlayer={meta.nlayer}, hs={meta.hs}, voc={meta.voc}")
    
    print("Step 4: Calling C API to create model...")
    from ctypes import c_int
    device_ids = (c_int * 1)(0)
    
    model = llaisys.libllaisys.LIB_LLAISYS.llaisysQwen2ModelCreate(
        byref(meta),
        0,  # CPU device
        device_ids,
        1
    )
    
    print(f"Step 5: Model created: {model}")
    
    if model:
        print("Step 6: Getting weights structure...")
        weights = llaisys.libllaisys.LIB_LLAISYS.llaisysQwen2ModelWeights(model)
        print(f"Step 7: Weights pointer: {weights}")
        
        print("Step 8: Destroying model...")
        llaisys.libllaisys.LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
        print("✓ All steps passed!")
    else:
        print("✗ Model creation failed!")
        
except Exception as e:
    print(f"✗ Error at current step: {e}")
    import traceback
    traceback.print_exc()


