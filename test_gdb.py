#!/usr/bin/env python
"""Simple test for gdb"""
import sys
print("Step 1: Importing llaisys")
sys.stdout.flush()

from llaisys.models.qwen2 import Qwen2
print("Step 2: Class imported")
sys.stdout.flush()

print("Step 3: About to create instance")
sys.stdout.flush()

m = Qwen2('models/DeepSeek-R1-Distill-Qwen-1.5B')
print("Step 4: Instance created")
sys.stdout.flush()


