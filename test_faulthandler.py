#!/usr/bin/env python
"""Test with faulthandler to get stack trace"""
import sys
import faulthandler

# Enable faulthandler to get stack trace on segfault
faulthandler.enable()

print("=" * 60)
print("Testing with faulthandler enabled")
print("=" * 60)

print("\n1. Importing llaisys.models.qwen2...")
sys.stdout.flush()

from llaisys.models.qwen2 import Qwen2

print("✓ Import successful")
print("\n2. Creating instance...")
sys.stdout.flush()

m = Qwen2('models/DeepSeek-R1-Distill-Qwen-1.5B')

print("✓ Instance created!")


