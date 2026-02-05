#!/usr/bin/env python
"""Test LLAISYS model only (skip PyTorch comparison)"""
import sys
import llaisys

print("=" * 60)
print("Testing LLAISYS Qwen2 Model")
print("=" * 60)

# Load model
print("\n1. Loading LLAISYS model...")
try:
    model = llaisys.models.Qwen2(
        "models/DeepSeek-R1-Distill-Qwen-1.5B",
        device=llaisys.DeviceType.CPU
    )
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Simple test tokens
print("\n2. Testing simple generation...")
test_tokens = [151644, 151645, 151646]  # Simple test
try:
    result = model.generate(test_tokens, max_new_tokens=5, top_k=1)
    print(f"✓ Input tokens: {test_tokens}")
    print(f"✓ Output tokens: {result}")
    print(f"✓ Generated {len(result) - len(test_tokens)} new tokens")
except Exception as e:
    print(f"✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ Basic test passed!")
print("=" * 60)

