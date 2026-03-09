from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys.qwen2 import LlaisysQwen2Meta, llaisysQwen2Model_t
from ..libllaisys import DeviceType
from ..libllaisys.llaisys_types import DataType
from ..tensor import Tensor
from ctypes import c_int, c_int64, c_size_t, POINTER, byref

from pathlib import Path
import safetensors
import json
import numpy as np
import torch


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        # Load config
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)

        # Extract model metadata
        nlayer = config["num_hidden_layers"]
        hs = config["hidden_size"]
        nh = config["num_attention_heads"]
        nkvh = config["num_key_value_heads"]
        dh = hs // nh
        di = config["intermediate_size"]
        maxseq = min(
            config.get("max_position_embeddings", 131072), 4096
        )  # Limit for memory
        voc = config["vocab_size"]
        epsilon = config["rms_norm_eps"]
        theta = config.get("rope_theta", 10000.0)
        end_token = config["eos_token_id"]

        # Determine dtype
        torch_dtype = config.get("torch_dtype", "bfloat16")
        if torch_dtype == "bfloat16":
            dtype = DataType.BF16
        elif torch_dtype == "float16":
            dtype = DataType.F16
        else:
            dtype = DataType.F32

        # Create metadata structure
        meta = LlaisysQwen2Meta()
        meta.dtype = dtype
        meta.nlayer = nlayer
        meta.hs = hs
        meta.nh = nh
        meta.nkvh = nkvh
        meta.dh = dh
        meta.di = di
        meta.maxseq = maxseq
        meta.voc = voc
        meta.epsilon = epsilon
        meta.theta = theta
        meta.end_token = end_token

        # Create model
        device_id = 0
        device_ids = (c_int * 1)(device_id)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta), device.value, device_ids, 1
        )

        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        self._config = config
        self._dtype = dtype
        self._device = device
        self._nlayer = nlayer

        # Keep references to all weight tensors to prevent premature deletion
        self._weight_tensors = []

        # Load weights from safetensors
        self._load_weights(model_path)

    def _load_weights(self, model_path):
        """Load weights from safetensors files
        Note: We use PyTorch ONLY for loading weight data from disk,
        NOT for model inference. All inference is done in C++ backend.
        """
        print(f"Loading weights from {model_path}...")

        weight_count = 0
        for file in sorted(model_path.glob("*.safetensors")):
            print(f"  Loading from {file.name}...")
            # Use PyTorch to load safetensors (supports bfloat16)
            import safetensors.torch

            weight_dict = safetensors.torch.load_file(str(file), device="cpu")
            print(f"    Found {len(weight_dict)} weights in file")

            for name, weight_torch in weight_dict.items():
                if weight_count % 10 == 0:
                    print(f"    Progress: {weight_count}/339 weights loaded...")
                # Convert to numpy for C++ loading (view as uint16 for bf16/fp16)
                if weight_torch.dtype == torch.bfloat16:
                    weight_np = weight_torch.view(torch.uint16).numpy()
                elif weight_torch.dtype == torch.float16:
                    weight_np = weight_torch.view(torch.int16).numpy().view(np.uint16)
                else:
                    weight_np = weight_torch.numpy()

                weight_count += 1

                # Map weight names to model structure
                if name == "model.embed_tokens.weight":
                    # Input embeddings: [vocab_size, hidden_size]
                    weight_np = np.ascontiguousarray(weight_np)
                    tensor = Tensor(
                        tuple(weight_np.shape), dtype=self._dtype, device=self._device
                    )
                    tensor.load(weight_np.ctypes.data)
                    LIB_LLAISYS.llaisysQwen2ModelSetInEmbed(self._model, tensor._tensor)
                    self._weight_tensors.append(tensor)  # Keep reference!

                elif name == "lm_head.weight":
                    # Output embeddings: [vocab_size, hidden_size]
                    weight_np = np.ascontiguousarray(weight_np)
                    tensor = Tensor(
                        tuple(weight_np.shape), dtype=self._dtype, device=self._device
                    )
                    tensor.load(weight_np.ctypes.data)
                    LIB_LLAISYS.llaisysQwen2ModelSetOutEmbed(
                        self._model, tensor._tensor
                    )
                    self._weight_tensors.append(tensor)

                elif name == "model.norm.weight":
                    # Final layer norm
                    weight_np = np.ascontiguousarray(weight_np)
                    tensor = Tensor(
                        tuple(weight_np.shape), dtype=self._dtype, device=self._device
                    )
                    tensor.load(weight_np.ctypes.data)
                    LIB_LLAISYS.llaisysQwen2ModelSetOutNormW(
                        self._model, tensor._tensor
                    )
                    self._weight_tensors.append(tensor)

                elif name.startswith("model.layers."):
                    # Parse layer index
                    parts = name.split(".")
                    layer_idx = int(parts[2])

                    if layer_idx >= self._nlayer:
                        continue

                    weight_name = ".".join(parts[3:])
                    weight_np = np.ascontiguousarray(weight_np)
                    tensor = Tensor(
                        tuple(weight_np.shape), dtype=self._dtype, device=self._device
                    )
                    tensor.load(weight_np.ctypes.data)

                    # Map to appropriate weight
                    if weight_name == "input_layernorm.weight":
                        LIB_LLAISYS.llaisysQwen2ModelSetLayerWeight(
                            self._model, b"attn_norm_w", layer_idx, tensor._tensor
                        )
                        self._weight_tensors.append(tensor)
                    elif weight_name == "self_attn.q_proj.weight":
                        LIB_LLAISYS.llaisysQwen2ModelSetLayerWeight(
                            self._model, b"attn_q_w", layer_idx, tensor._tensor
                        )
                        self._weight_tensors.append(tensor)
                    elif weight_name == "self_attn.q_proj.bias":
                        LIB_LLAISYS.llaisysQwen2ModelSetLayerWeight(
                            self._model, b"attn_q_b", layer_idx, tensor._tensor
                        )
                        self._weight_tensors.append(tensor)
                    elif weight_name == "self_attn.k_proj.weight":
                        LIB_LLAISYS.llaisysQwen2ModelSetLayerWeight(
                            self._model, b"attn_k_w", layer_idx, tensor._tensor
                        )
                        self._weight_tensors.append(tensor)
                    elif weight_name == "self_attn.k_proj.bias":
                        LIB_LLAISYS.llaisysQwen2ModelSetLayerWeight(
                            self._model, b"attn_k_b", layer_idx, tensor._tensor
                        )
                        self._weight_tensors.append(tensor)
                    elif weight_name == "self_attn.v_proj.weight":
                        LIB_LLAISYS.llaisysQwen2ModelSetLayerWeight(
                            self._model, b"attn_v_w", layer_idx, tensor._tensor
                        )
                        self._weight_tensors.append(tensor)
                    elif weight_name == "self_attn.v_proj.bias":
                        LIB_LLAISYS.llaisysQwen2ModelSetLayerWeight(
                            self._model, b"attn_v_b", layer_idx, tensor._tensor
                        )
                        self._weight_tensors.append(tensor)
                    elif weight_name == "self_attn.o_proj.weight":
                        LIB_LLAISYS.llaisysQwen2ModelSetLayerWeight(
                            self._model, b"attn_o_w", layer_idx, tensor._tensor
                        )
                        self._weight_tensors.append(tensor)
                    elif weight_name == "post_attention_layernorm.weight":
                        LIB_LLAISYS.llaisysQwen2ModelSetLayerWeight(
                            self._model, b"mlp_norm_w", layer_idx, tensor._tensor
                        )
                        self._weight_tensors.append(tensor)
                    elif weight_name == "mlp.gate_proj.weight":
                        LIB_LLAISYS.llaisysQwen2ModelSetLayerWeight(
                            self._model, b"mlp_gate_w", layer_idx, tensor._tensor
                        )
                        self._weight_tensors.append(tensor)
                    elif weight_name == "mlp.up_proj.weight":
                        LIB_LLAISYS.llaisysQwen2ModelSetLayerWeight(
                            self._model, b"mlp_up_w", layer_idx, tensor._tensor
                        )
                        self._weight_tensors.append(tensor)
                    elif weight_name == "mlp.down_proj.weight":
                        LIB_LLAISYS.llaisysQwen2ModelSetLayerWeight(
                            self._model, b"mlp_down_w", layer_idx, tensor._tensor
                        )
                        self._weight_tensors.append(tensor)

        print(f"Weights loaded successfully! Total: {weight_count} tensors")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """Generate tokens"""
        if max_new_tokens is None:
            max_new_tokens = 128

        # For now, only support greedy sampling (top_k=1)
        if top_k != 1:
            print("Warning: Only greedy sampling (top_k=1) is currently supported")

        output_tokens = list(inputs)
        import sys

        # print(f"[DEBUG] Starting generation with {len(output_tokens)} tokens")
        sys.stdout.flush()

        for i in range(max_new_tokens):
            # print(f"[DEBUG] Step {i+1}/{max_new_tokens}")
            sys.stdout.flush()

            # Convert to ctypes array
            token_array = (c_int64 * len(output_tokens))(*output_tokens)

            # print(f"[DEBUG] Calling C++ infer with seq_len={len(output_tokens)}")
            sys.stdout.flush()

            # Call inference
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, token_array, len(output_tokens)
            )

            # print(f"[DEBUG] Got next_token={next_token}")
            sys.stdout.flush()

            # Check for end token
            if next_token == self._config["eos_token_id"]:
                # Match HuggingFace `generate()` behavior: include eos token in returned sequence.
                output_tokens.append(int(next_token))
                break

            output_tokens.append(int(next_token))

        return output_tokens

    def __del__(self):
        """Cleanup"""
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
