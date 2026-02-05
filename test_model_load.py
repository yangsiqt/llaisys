#!/usr/bin/env python
"""Test model loading"""
import llaisys

print("Testing Qwen2 model loading...")
print("Creating model...")

try:
    model = llaisys.models.Qwen2(
        "models/DeepSeek-R1-Distill-Qwen-1.5B",
        device=llaisys.DeviceType.CPU
    )
    print("✓ Model created and weights loaded successfully!")
    
    # Test simple generation
    print("\nTesting generation with simple input...")
    test_tokens = [151644, 151645, 151646, 151643]  # Simple test tokens
    result = model.generate(test_tokens, max_new_tokens=5, top_k=1)
    print(f"✓ Generation works! Result: {result}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

