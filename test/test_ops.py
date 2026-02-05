#!/usr/bin/env python
"""
Test all operators
"""
import sys
import os
import argparse

# Add test directory to path
test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, test_dir)

# Import all operator tests
from ops import argmax, embedding, linear, rms_norm, rope, self_attention, swiglu

def run_all_tests(device="cpu", profile=False):
    """Run all operator tests"""
    
    print(f"\n{'='*60}")
    print(f"Running all operator tests on {device}")
    print(f"{'='*60}\n")
    
    # Test argmax
    print("\n--- Testing Argmax ---")
    test_shapes = [(4,), (4096,)]
    test_dtypes = ["f32", "f16", "bf16"]
    for shape in test_shapes:
        for dtype in test_dtypes:
            argmax.test_op_argmax(shape, dtype, device, profile)
    
    # Test embedding
    print("\n--- Testing Embedding ---")
    test_shapes = [
        ((1,), (2, 3)),
        ((50,), (512, 4096)),
    ]
    for idx_shape, embd_shape in test_shapes:
        for dtype in test_dtypes:
            embedding.test_op_embedding(idx_shape, embd_shape, dtype, device, profile)
    
    # Test linear
    print("\n--- Testing Linear ---")
    test_shapes = [
        ((2, 3), (2, 4), (3, 4), True),
        ((512, 4096), (512, 4096), (4096, 4096), True),
    ]
    for out_shape, x_shape, w_shape, use_bias in test_shapes:
        for dtype in test_dtypes:
            if dtype == "f32":
                atol, rtol = 1e-5, 1e-5
            else:
                atol, rtol = 1e-2, 1e-2
            linear.test_op_linear(out_shape, x_shape, w_shape, use_bias, dtype, atol, rtol, device, profile)
    
    # Test rms_norm
    print("\n--- Testing RMS Norm ---")
    test_shapes = [(1, 4), (512, 4096)]
    for shape in test_shapes:
        for dtype in test_dtypes:
            if dtype == "f32":
                atol, rtol = 1e-5, 1e-5
            else:
                atol, rtol = 1e-2, 1e-2
            rms_norm.test_op_rms_norm(shape, dtype, atol, rtol, device, profile)
    
    # Test rope
    print("\n--- Testing RoPE ---")
    test_shapes = [
        ((2, 1, 4), (0, 2)),
        ((512, 4, 4096), (512, 1024)),
    ]
    for shape, start_end in test_shapes:
        for dtype in test_dtypes:
            if dtype == "f32":
                atol, rtol = 1e-4, 1e-4
            else:
                atol, rtol = 1e-2, 1e-2
            rope.test_op_rope(shape, start_end, dtype, atol, rtol, device, profile)
    
    # Test self_attention
    print("\n--- Testing Self-Attention ---")
    test_configs = [
        (2, 2, 1, 1, 4),
        (5, 11, 4, 2, 8),
    ]
    for qlen, kvlen, nh, nkvh, hd in test_configs:
        for dtype in test_dtypes:
            if dtype == "f32":
                atol, rtol = 1e-5, 1e-5
            else:
                atol, rtol = 1e-2, 1e-2
            self_attention.test_op_self_attention(qlen, kvlen, nh, nkvh, hd, dtype, atol, rtol, device, profile)
    
    # Test swiglu
    print("\n--- Testing SwiGLU ---")
    test_shapes = [(2, 3), (512, 4096)]
    for shape in test_shapes:
        for dtype in test_dtypes:
            if dtype == "f32":
                atol, rtol = 1e-5, 1e-5
            else:
                atol, rtol = 1e-2, 1e-2
            swiglu.test_op_swiglu(shape, dtype, atol, rtol, device, profile)
    
    print(f"\n{'='*60}")
    print("\033[92mAll operator tests passed!\033[0m")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    
    run_all_tests(args.device, args.profile)

