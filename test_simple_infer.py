#!/usr/bin/env python
"""Simple inference test"""
import llaisys
import sys

print("=" * 60)
print("Simple LLAISYS Inference Test")
print("=" * 60)

# Load model
print("\n1. Loading model...")
sys.stdout.flush()
model = llaisys.models.Qwen2("models/DeepSeek-R1-Distill-Qwen-1.5B")
print("✓ Model loaded")

# Test simple inference
print("\n2. Running inference...")
sys.stdout.flush()
test_tokens = [151644, 151645, 151646]  # Simple token IDs
print(f"   Input tokens: {test_tokens}")
sys.stdout.flush()

try:
    result = model.generate(test_tokens, max_new_tokens=3, top_k=1)
    print(f"✓ Output tokens: {result}")
    print(f"✓ Generated {len(result) - len(test_tokens)} new tokens")
except Exception as e:
    print(f"✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ Test passed!")
print("=" * 60)


