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

    def generate_with_pd_metrics_batch(
        self,
        inputs_batch,
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        import sys
        import time

        if max_new_tokens is None:
            max_new_tokens = 128
        if top_k != 1:
            print("Warning: Only greedy sampling (top_k=1) is currently supported")

        bs = len(inputs_batch)
        max_prompt_len = max(len(p) for p in inputs_batch)

        padded = []
        for p in inputs_batch:
            pad_token = p[0] if p else 0
            padded_p = [pad_token] * (max_prompt_len - len(p)) + p
            padded.append(padded_p)

        outputs = [list(p) for p in padded]
        n_prompt_per_seq = max_prompt_len

        t_wall0 = time.perf_counter()
        prefill_s = 0.0
        sum_decode_kernel_s = 0.0
        n_gen = 0

        for step in range(max_new_tokens):
            current_len = n_prompt_per_seq + step
            token_array = (c_int64 * (bs * current_len))()
            for b in range(bs):
                base = b * current_len
                for i, tok in enumerate(outputs[b]):
                    token_array[base + i] = tok

            t0 = time.perf_counter()
            next_arr = (c_int64 * bs)()
            LIB_LLAISYS.llaisysQwen2ModelInferBatch(
                self._model, token_array, bs * current_len, next_arr, bs
            )
            dt = time.perf_counter() - t0

            if step == 0:
                prefill_s = dt / bs
            else:
                sum_decode_kernel_s += dt

            done = [False] * bs
            for b in range(bs):
                if next_arr[b] == self._config["eos_token_id"]:
                    outputs[b].append(int(next_arr[b]))
                    done[b] = True
                else:
                    outputs[b].append(int(next_arr[b]))

            n_gen += 1
            if all(done):
                break

            if n_gen > 0 and n_gen % 10 == 0:
                sys.stdout.write(f"\r[Batch] Generated {n_gen}/{max_new_tokens} tokens...")
                sys.stdout.flush()

        print()
        t_wall1 = time.perf_counter()
        total_s = t_wall1 - t_wall0
        decode_wall_s = max(0.0, total_s - prefill_s)

        final_outputs = []
        for b in range(bs):
            final_outputs.append(outputs[b][max_prompt_len - len(inputs_batch[b]):])

        metrics = {
            "n_prompt": n_prompt_per_seq,
            "n_generated": n_gen,
            "batch_size": bs,
            "prefill_s": prefill_s,
            "decode_wall_s": decode_wall_s,
            "total_s": total_s,
            "sum_decode_kernel_s": sum_decode_kernel_s,
        }
        return final_outputs, metrics

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

    def generate_with_pd_metrics(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """与 generate 相同逻辑，额外返回 prefill / decode 分段耗时（用于 benchmark）。"""
        import sys
        import time

        if max_new_tokens is None:
            max_new_tokens = 128
        if top_k != 1:
            print("Warning: Only greedy sampling (top_k=1) is currently supported")

        output_tokens = list(inputs)
        n_prompt = len(output_tokens)

        t_wall0 = time.perf_counter()
        prefill_s = 0.0
        sum_decode_kernel_s = 0.0

        for i in range(max_new_tokens):
            token_array = (c_int64 * len(output_tokens))(*output_tokens)
            t0 = time.perf_counter()
            next_token = LIB_LLAISYS.llaisysQwen2TPModelInfer(
                self._model, token_array, len(output_tokens)
            )
            dt = time.perf_counter() - t0
            if i == 0:
                prefill_s = dt
            else:
                sum_decode_kernel_s += dt
            if next_token == self._config["eos_token_id"]:
                output_tokens.append(int(next_token))
                break
            output_tokens.append(int(next_token))
            if i > 0 and i % 10 == 0:
                sys.stdout.write(f"\r[TP] Generated {i}/{max_new_tokens} tokens...")
                sys.stdout.flush()

        print()
        t_wall1 = time.perf_counter()
        n_gen = len(output_tokens) - n_prompt
        total_s = t_wall1 - t_wall0
        decode_wall_s = max(0.0, total_s - prefill_s)

        metrics = {
            "n_prompt": n_prompt,
            "n_generated": n_gen,
            "prefill_s": prefill_s,
            "decode_wall_s": decode_wall_s,
            "total_s": total_s,
            "sum_decode_kernel_s": sum_decode_kernel_s,
        }

        return output_tokens, metrics

    def generate_with_pd_metrics_batch(
        self,
        inputs_batch,
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """Batch generate: inputs_batch is a list of token lists.
        All sequences are padded to the same length, then decoded in lockstep.
        Returns (list of output token lists, metrics dict)."""
        import sys
        import time

        if max_new_tokens is None:
            max_new_tokens = 128
        if top_k != 1:
            print("Warning: Only greedy sampling (top_k=1) is currently supported")

        bs = len(inputs_batch)
        max_prompt_len = max(len(p) for p in inputs_batch)

        # Pad all prompts to the same length
        padded = []
        for p in inputs_batch:
            # Left-pad with a dummy token (use first token as pad)
            pad_token = p[0] if p else 0
            padded_p = [pad_token] * (max_prompt_len - len(p)) + p
            padded.append(padded_p)

        # Flatten to [bs * max_prompt_len] for C++ prefill
        flat_tokens = []
        for p in padded:
            flat_tokens.extend(p)

        # Track per-sequence outputs
        outputs = [list(p) for p in padded]  # real tokens (no pad)

        n_prompt_per_seq = max_prompt_len
        t_wall0 = time.perf_counter()
        prefill_s = 0.0
        sum_decode_kernel_s = 0.0

        n_gen = 0

        for step in range(max_new_tokens):
            # Build flat token array: [bs * (n_prompt + step)]
            current_len = n_prompt_per_seq + step
            token_array = (c_int64 * (bs * current_len))()
            for b in range(bs):
                base = b * current_len
                for i, tok in enumerate(outputs[b]):
                    token_array[base + i] = tok

            t0 = time.perf_counter()
            next_arr = (c_int64 * bs)()
            LIB_LLAISYS.llaisysQwen2TPModelInferBatch(
                self._model, token_array, bs * current_len, next_arr, bs
            )
            dt = time.perf_counter() - t0

            if step == 0:
                prefill_s = dt / bs  # per-sequence prefill amortized
            else:
                sum_decode_kernel_s += dt

            # Append tokens
            done = [False] * bs
            for b in range(bs):
                if next_arr[b] == self._config["eos_token_id"]:
                    outputs[b].append(int(next_arr[b]))
                    done[b] = True
                else:
                    outputs[b].append(int(next_arr[b]))

            n_gen += 1
            if all(done):
                break

            if n_gen > 0 and n_gen % 10 == 0:
                sys.stdout.write(f"\r[TP Batch] Generated {n_gen}/{max_new_tokens} tokens...")
                sys.stdout.flush()

        print()
        t_wall1 = time.perf_counter()
        total_s = t_wall1 - t_wall0
        decode_wall_s = max(0.0, total_s - prefill_s)

        # Strip padding from outputs
        final_outputs = []
        for b in range(bs):
            final_outputs.append(outputs[b][max_prompt_len - len(inputs_batch[b]):])

        metrics = {
            "n_prompt": n_prompt_per_seq,
            "n_generated": n_gen,
            "batch_size": bs,
            "prefill_s": prefill_s,
            "decode_wall_s": decode_wall_s,
            "total_s": total_s,
            "sum_decode_kernel_s": sum_decode_kernel_s,
        }

        return final_outputs, metrics

    def init_continuous(self, max_slots: int):
        rc = LIB_LLAISYS.llaisysQwen2TPModelInitContinuous(self._model, max_slots)
        if rc != 0:
            raise RuntimeError("llaisysQwen2TPModelInitContinuous failed")

    def prefill_slot(self, slot_id: int, token_ids: Sequence[int]) -> int:
        token_array = (c_int64 * len(token_ids))(*token_ids)
        next_token = LIB_LLAISYS.llaisysQwen2TPModelPrefillSlot(
            self._model, slot_id, token_array, len(token_ids)
        )
        if next_token < 0:
            raise RuntimeError("llaisysQwen2TPModelPrefillSlot failed")
        return int(next_token)

    def prefill_slots(self, slot_ids: Sequence[int], inputs_batch):
        if not slot_ids:
            return []
        if len(slot_ids) != len(inputs_batch):
            raise ValueError("slot_ids and inputs_batch must have the same length")
        prompt_len = len(inputs_batch[0])
        if prompt_len == 0:
            raise ValueError("prompt_len must be > 0")
        flat_tokens = []
        for tokens in inputs_batch:
            if len(tokens) != prompt_len:
                raise ValueError("prefill_slots requires same-length prompts")
            flat_tokens.extend(tokens)
        slot_array = (c_size_t * len(slot_ids))(*slot_ids)
        token_array = (c_int64 * len(flat_tokens))(*flat_tokens)
        out_array = (c_int64 * len(slot_ids))()
        rc = LIB_LLAISYS.llaisysQwen2TPModelPrefillSlots(
            self._model, slot_array, token_array, out_array, len(slot_ids), prompt_len
        )
        if rc != 0:
            raise RuntimeError("llaisysQwen2TPModelPrefillSlots failed")
        return [int(out_array[i]) for i in range(len(slot_ids))]

    def decode_slots(self, slot_ids: Sequence[int], input_tokens: Sequence[int]):
        if len(slot_ids) != len(input_tokens):
            raise ValueError("slot_ids and input_tokens must have the same length")
        if not slot_ids:
            return []
        slot_array = (c_size_t * len(slot_ids))(*slot_ids)
        token_array = (c_int64 * len(input_tokens))(*input_tokens)
        out_array = (c_int64 * len(slot_ids))()
        rc = LIB_LLAISYS.llaisysQwen2TPModelDecodeSlots(
            self._model, slot_array, token_array, out_array, len(slot_ids)
        )
        if rc != 0:
            raise RuntimeError("llaisysQwen2TPModelDecodeSlots failed")
        return [int(out_array[i]) for i in range(len(slot_ids))]

    def release_slot(self, slot_id: int):
        rc = LIB_LLAISYS.llaisysQwen2TPModelReleaseSlot(self._model, slot_id)
        if rc != 0:
            raise RuntimeError("llaisysQwen2TPModelReleaseSlot failed")

    def slot_seq_len(self, slot_id: int) -> int:
        return int(LIB_LLAISYS.llaisysQwen2TPModelSlotSeqLen(self._model, slot_id))

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2TPModelDestroy(self._model)


class Qwen2TPContinuousEngine:
    """Fixed-slot continuous batching helper for online-serving benchmarks."""

    def __init__(self, model: Qwen2TP, max_slots: int, eos_token_id=None, ignore_eos: bool = True):
        self.model = model
        self.max_slots = int(max_slots)
        self.eos_token_id = eos_token_id if eos_token_id is not None else model._config.get("eos_token_id")
        self.ignore_eos = ignore_eos
        self.free_slots = list(range(self.max_slots))
        self.active = {}
        self.stats = {
            "prefill_calls": 0,
            "prefill_batches": 0,
            "prefill_s": 0.0,
            "prefill_tokens": 0,
            "decode_steps": 0,
            "decode_s": 0.0,
            "decode_slot_steps": 0,
        }
        self.model.init_continuous(self.max_slots)

    def can_accept(self) -> bool:
        return bool(self.free_slots)

    def submit(self, request_id: int, prompt_tokens: Sequence[int], target_output_len: int, now_s: float):
        if not self.free_slots:
            return None
        import time
        slot_id = self.free_slots.pop(0)
        t0 = time.perf_counter()
        first_token = self.model.prefill_slot(slot_id, prompt_tokens)
        prefill_dt = time.perf_counter() - t0
        first_token_s = time.perf_counter()
        self.stats["prefill_calls"] += 1
        self.stats["prefill_batches"] += 1
        self.stats["prefill_s"] += prefill_dt
        self.stats["prefill_tokens"] += len(prompt_tokens)
        state = {
            "request_id": request_id,
            "slot_id": slot_id,
            "prompt_len": len(prompt_tokens),
            "target_output_len": int(target_output_len),
            "arrival_s": now_s,
            "first_token_s": first_token_s,
            "finish_s": None,
            "output_tokens": [],
            "last_token": first_token,
        }
        state["output_tokens"].append(first_token)
        self.active[slot_id] = state
        if self._is_finished(state, first_token):
            return self._finish_slot(slot_id, first_token_s)
        return None

    def submit_many_same_len(self, requests, now_s: float):
        if not requests:
            return []
        import time
        if len(requests) > len(self.free_slots):
            raise ValueError("not enough free slots for submit_many_same_len")
        prompt_len = len(requests[0]["prompt_tokens"])
        for req in requests:
            if len(req["prompt_tokens"]) != prompt_len:
                raise ValueError("submit_many_same_len requires same-length prompts")

        slot_ids = [self.free_slots.pop(0) for _ in requests]
        t0 = time.perf_counter()
        first_tokens = self.model.prefill_slots(slot_ids, [r["prompt_tokens"] for r in requests])
        prefill_dt = time.perf_counter() - t0
        first_token_s = time.perf_counter()
        completed = []
        self.stats["prefill_calls"] += 1
        self.stats["prefill_batches"] += len(requests)
        self.stats["prefill_s"] += prefill_dt
        self.stats["prefill_tokens"] += prompt_len * len(requests)

        for slot_id, req, first_token in zip(slot_ids, requests, first_tokens):
            state = {
                "request_id": req["request_id"],
                "slot_id": slot_id,
                "prompt_len": len(req["prompt_tokens"]),
                "target_output_len": int(req["output_len"]),
                "arrival_s": now_s if "arrival_abs_s" not in req else req["arrival_abs_s"],
                "first_token_s": first_token_s,
                "finish_s": None,
                "output_tokens": [first_token],
                "last_token": first_token,
            }
            self.active[slot_id] = state
            if self._is_finished(state, first_token):
                completed.append(self._finish_slot(slot_id, first_token_s))
        return [c for c in completed if c is not None]

    def step(self, now_s: float):
        if not self.active:
            return []

        completed = []
        slot_ids = sorted(self.active)
        input_tokens = [self.active[s]["last_token"] for s in slot_ids]
        import time
        t0 = time.perf_counter()
        next_tokens = self.model.decode_slots(slot_ids, input_tokens)
        self.stats["decode_s"] += time.perf_counter() - t0
        self.stats["decode_steps"] += 1
        self.stats["decode_slot_steps"] += len(slot_ids)
        for slot_id, next_token in zip(slot_ids, next_tokens):
            if slot_id not in self.active:
                continue
            state = self.active[slot_id]
            state["last_token"] = next_token
            state["output_tokens"].append(next_token)
            if self._is_finished(state, next_token):
                completed.append(self._finish_slot(slot_id, now_s))
        return [c for c in completed if c is not None]

    def active_count(self) -> int:
        return len(self.active)

    def metrics(self):
        return dict(self.stats)

    def _is_finished(self, state, token: int) -> bool:
        if len(state["output_tokens"]) >= state["target_output_len"]:
            return True
        if not self.ignore_eos and self.eos_token_id is not None and token == self.eos_token_id:
            return True
        return False

    def _finish_slot(self, slot_id: int, now_s: float):
        state = self.active.pop(slot_id, None)
        if state is None:
            return None
        state["finish_s"] = now_s
        self.model.release_slot(slot_id)
        self.free_slots.append(slot_id)
        self.free_slots.sort()
        return state
