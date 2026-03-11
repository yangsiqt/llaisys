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


class Qwen2TP:
    """Tensor-parallel Qwen2 model across multiple NVIDIA GPUs."""

    def __init__(self, model_path, device: DeviceType = DeviceType.NVIDIA,
                 device_ids=None):
        model_path = Path(model_path)

        if device_ids is None:
            device_ids = [0, 1]
        self._device_ids = device_ids
        self._tp_size = len(device_ids)

        with open(model_path / "config.json", "r") as f:
            config = json.load(f)

        nlayer = config["num_hidden_layers"]
        hs = config["hidden_size"]
        nh = config["num_attention_heads"]
        nkvh = config["num_key_value_heads"]
        dh = hs // nh
        di = config["intermediate_size"]
        maxseq = min(config.get("max_position_embeddings", 131072), 4096)
        voc = config["vocab_size"]
        epsilon = config["rms_norm_eps"]
        theta = config.get("rope_theta", 10000.0)
        end_token = config["eos_token_id"]

        torch_dtype = config.get("torch_dtype", "bfloat16")
        if torch_dtype == "bfloat16":
            dtype = DataType.BF16
        elif torch_dtype == "float16":
            dtype = DataType.F16
        else:
            dtype = DataType.F32

        assert nh % self._tp_size == 0, f"nh({nh}) must be divisible by tp_size({self._tp_size})"
        assert nkvh % self._tp_size == 0, f"nkvh({nkvh}) must be divisible by tp_size({self._tp_size})"
        assert di % self._tp_size == 0, f"di({di}) must be divisible by tp_size({self._tp_size})"

        self._nh_per_rank = nh // self._tp_size
        self._nkvh_per_rank = nkvh // self._tp_size
        self._di_per_rank = di // self._tp_size

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

        c_device_ids = (c_int * len(device_ids))(*device_ids)
        self._model = LIB_LLAISYS.llaisysQwen2TPModelCreate(
            byref(meta), device.value, c_device_ids, len(device_ids)
        )

        if not self._model:
            raise RuntimeError("Failed to create Qwen2TP model")

        self._config = config
        self._dtype = dtype
        self._device = device
        self._nlayer = nlayer
        self._weight_tensors = []

        self._load_weights(model_path)

    def _make_tensor(self, weight_np, device_id):
        """Create a Tensor on the given GPU and load numpy data into it."""
        weight_np = np.ascontiguousarray(weight_np)
        tensor = Tensor(
            tuple(weight_np.shape), dtype=self._dtype,
            device=self._device, device_id=device_id
        )
        tensor.load(weight_np.ctypes.data)
        self._weight_tensors.append(tensor)
        return tensor

    def _to_np(self, weight_torch):
        """Convert a torch tensor to numpy (handling bf16/fp16)."""
        if weight_torch.dtype == torch.bfloat16:
            return weight_torch.view(torch.uint16).numpy()
        elif weight_torch.dtype == torch.float16:
            return weight_torch.view(torch.int16).numpy().view(np.uint16)
        else:
            return weight_torch.numpy()

    def _load_weights(self, model_path):
        import sys
        print(f"[TP] Loading weights from {model_path} with tp_size={self._tp_size} ...", flush=True)

        weight_count = 0
        for file in sorted(model_path.glob("*.safetensors")):
            print(f"  Loading from {file.name}...", flush=True)
            import safetensors.torch
            weight_dict = safetensors.torch.load_file(str(file), device="cpu")

            for name, weight_torch in weight_dict.items():
                weight_count += 1
                if weight_count % 20 == 0:
                    print(f"    Progress: {weight_count} weights loaded...", flush=True)

                if name == "model.embed_tokens.weight":
                    weight_np = self._to_np(weight_torch)
                    for rank in range(self._tp_size):
                        t = self._make_tensor(weight_np, self._device_ids[rank])
                        LIB_LLAISYS.llaisysQwen2TPModelSetInEmbed(
                            self._model, rank, t._tensor)

                elif name == "lm_head.weight":
                    weight_np = self._to_np(weight_torch)
                    t = self._make_tensor(weight_np, self._device_ids[0])
                    LIB_LLAISYS.llaisysQwen2TPModelSetOutEmbed(
                        self._model, t._tensor)

                elif name == "model.norm.weight":
                    weight_np = self._to_np(weight_torch)
                    for rank in range(self._tp_size):
                        t = self._make_tensor(weight_np, self._device_ids[rank])
                        LIB_LLAISYS.llaisysQwen2TPModelSetOutNormW(
                            self._model, rank, t._tensor)

                elif name.startswith("model.layers."):
                    parts = name.split(".")
                    layer_idx = int(parts[2])
                    if layer_idx >= self._nlayer:
                        continue
                    weight_name = ".".join(parts[3:])
                    self._load_layer_weight(layer_idx, weight_name, weight_torch)

        # Handle tied embeddings
        if not any("lm_head.weight" in str(f) for f in model_path.glob("*.safetensors")):
            pass

        print(f"[TP] Weights loaded successfully! Total: {weight_count} tensors")

    def _load_layer_weight(self, layer_idx, weight_name, weight_torch):
        tp = self._tp_size

        # Column-parallel: split output dim (dim 0 of weight matrix)
        col_parallel_map = {
            "self_attn.q_proj.weight": ("attn_q_w", self._nh_per_rank * (self._config["hidden_size"] // self._config["num_attention_heads"])),
            "self_attn.q_proj.bias":   ("attn_q_b", None),
            "self_attn.k_proj.weight": ("attn_k_w", self._nkvh_per_rank * (self._config["hidden_size"] // self._config["num_attention_heads"])),
            "self_attn.k_proj.bias":   ("attn_k_b", None),
            "self_attn.v_proj.weight": ("attn_v_w", self._nkvh_per_rank * (self._config["hidden_size"] // self._config["num_attention_heads"])),
            "self_attn.v_proj.bias":   ("attn_v_b", None),
            "mlp.gate_proj.weight":    ("mlp_gate_w", self._di_per_rank),
            "mlp.up_proj.weight":      ("mlp_up_w", self._di_per_rank),
        }

        # Row-parallel: split input dim (dim 1 of weight matrix)
        row_parallel_map = {
            "self_attn.o_proj.weight": "attn_o_w",
            "mlp.down_proj.weight":    "mlp_down_w",
        }

        # Replicated weights
        replicated_map = {
            "input_layernorm.weight":          "attn_norm_w",
            "post_attention_layernorm.weight":  "mlp_norm_w",
        }

        if weight_name in col_parallel_map:
            api_name, _ = col_parallel_map[weight_name]
            shards = weight_torch.chunk(tp, dim=0)
            for rank, shard in enumerate(shards):
                weight_np = self._to_np(shard.contiguous())
                t = self._make_tensor(weight_np, self._device_ids[rank])
                LIB_LLAISYS.llaisysQwen2TPModelSetLayerWeight(
                    self._model, rank, api_name.encode(), layer_idx, t._tensor)

        elif weight_name in row_parallel_map:
            api_name = row_parallel_map[weight_name]
            shards = weight_torch.chunk(tp, dim=1)
            for rank, shard in enumerate(shards):
                weight_np = self._to_np(shard.contiguous())
                t = self._make_tensor(weight_np, self._device_ids[rank])
                LIB_LLAISYS.llaisysQwen2TPModelSetLayerWeight(
                    self._model, rank, api_name.encode(), layer_idx, t._tensor)

        elif weight_name in replicated_map:
            api_name = replicated_map[weight_name]
            weight_np = self._to_np(weight_torch)
            for rank in range(tp):
                t = self._make_tensor(weight_np, self._device_ids[rank])
                LIB_LLAISYS.llaisysQwen2TPModelSetLayerWeight(
                    self._model, rank, api_name.encode(), layer_idx, t._tensor)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128

        if top_k != 1:
            print("Warning: Only greedy sampling (top_k=1) is currently supported")

        output_tokens = list(inputs)
        import sys

        for i in range(max_new_tokens):
            token_array = (c_int64 * len(output_tokens))(*output_tokens)

            next_token = LIB_LLAISYS.llaisysQwen2TPModelInfer(
                self._model, token_array, len(output_tokens)
            )

            if next_token == self._config["eos_token_id"]:
                output_tokens.append(int(next_token))
                break

            output_tokens.append(int(next_token))

            if (i + 1) % 10 == 0:
                sys.stdout.write(f"\r[TP] Generated {i+1}/{max_new_tokens} tokens...")
                sys.stdout.flush()

        print()
        return output_tokens

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2TPModelDestroy(self._model)
