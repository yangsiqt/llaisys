#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/kv_cache_nvidia.hpp"
#endif

namespace llaisys::ops {
void scatter_kv_decode(tensor_t k_cache, tensor_t v_cache, tensor_t k, tensor_t v,
                       tensor_t slot_ids, tensor_t positions) {
    CHECK_SAME_DEVICE(k_cache, v_cache, k, v, slot_ids, positions);
    CHECK_SAME_DTYPE(k_cache->dtype(), v_cache->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(k_cache->ndim() == 4 && v_cache->ndim() == 4, "KV cache must be 4D");
    CHECK_ARGUMENT(k->ndim() == 3 && v->ndim() == 3, "K/V must be 3D [batch,nkvh,dh]");
    CHECK_ARGUMENT(slot_ids->dtype() == LLAISYS_DTYPE_I64 && positions->dtype() == LLAISYS_DTYPE_I64,
                   "slot_ids/positions must be int64");
    CHECK_ARGUMENT(k_cache->isContiguous() && v_cache->isContiguous() && k->isContiguous() &&
                       v->isContiguous() && slot_ids->isContiguous() && positions->isContiguous(),
                   "scatter_kv_decode expects contiguous tensors");
    CHECK_ARGUMENT(k_cache->shape() == v_cache->shape(), "K/V cache shape mismatch");
    CHECK_ARGUMENT(k->shape() == v->shape(), "K/V shape mismatch");
    CHECK_ARGUMENT(k->shape()[1] == k_cache->shape()[2], "nkvh mismatch");
    CHECK_ARGUMENT(k->shape()[2] == k_cache->shape()[3], "head_dim mismatch");
    CHECK_ARGUMENT(slot_ids->shape()[0] == k->shape()[0] && positions->shape()[0] == k->shape()[0],
                   "metadata batch mismatch");

    llaisys::core::context().setDevice(k_cache->deviceType(), k_cache->deviceId());
    switch (k_cache->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::scatter_kv_decode(k_cache->data(), v_cache->data(), k->data(), v->data(),
                                         reinterpret_cast<const int64_t *>(slot_ids->data()),
                                         reinterpret_cast<const int64_t *>(positions->data()), k_cache->dtype(),
                                         k->shape()[0], k_cache->shape()[1], k_cache->shape()[2],
                                         k_cache->shape()[3]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void rope_and_scatter_kv_decode(tensor_t q, tensor_t k, tensor_t v,
                                tensor_t k_cache, tensor_t v_cache,
                                tensor_t pos_ids, tensor_t slot_ids,
                                tensor_t positions, float theta) {
    CHECK_SAME_DEVICE(q, k, v, k_cache, v_cache, pos_ids, slot_ids, positions);
    CHECK_SAME_DTYPE(q->dtype(), k->dtype(), v->dtype(), k_cache->dtype(), v_cache->dtype());
    CHECK_ARGUMENT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
                   "Q/K/V must be 3D [batch,nheads,dh]");
    CHECK_ARGUMENT(k_cache->ndim() == 4 && v_cache->ndim() == 4, "KV cache must be 4D");
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64 && slot_ids->dtype() == LLAISYS_DTYPE_I64 &&
                       positions->dtype() == LLAISYS_DTYPE_I64,
                   "pos_ids/slot_ids/positions must be int64");
    CHECK_ARGUMENT(q->isContiguous() && k->isContiguous() && v->isContiguous() &&
                       k_cache->isContiguous() && v_cache->isContiguous() &&
                       pos_ids->isContiguous() && slot_ids->isContiguous() && positions->isContiguous(),
                   "rope_and_scatter_kv_decode expects contiguous tensors");
    CHECK_ARGUMENT(k_cache->shape() == v_cache->shape(), "K/V cache shape mismatch");
    CHECK_ARGUMENT(k->shape() == v->shape(), "K/V shape mismatch");
    CHECK_ARGUMENT(q->shape()[0] == k->shape()[0], "Q/K batch mismatch");
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2] && q->shape()[2] == v->shape()[2], "head_dim mismatch");
    CHECK_ARGUMENT(k->shape()[1] == k_cache->shape()[2], "nkvh mismatch");
    CHECK_ARGUMENT(k->shape()[2] == k_cache->shape()[3], "cache head_dim mismatch");
    CHECK_ARGUMENT(pos_ids->shape()[0] == q->shape()[0] && slot_ids->shape()[0] == q->shape()[0] &&
                       positions->shape()[0] == q->shape()[0],
                   "metadata batch mismatch");
    CHECK_ARGUMENT(q->shape()[2] % 2 == 0, "RoPE head_dim must be even");

    llaisys::core::context().setDevice(k_cache->deviceType(), k_cache->deviceId());
    switch (k_cache->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope_and_scatter_kv_decode(
            q->data(), k->data(), v->data(), k_cache->data(), v_cache->data(),
            reinterpret_cast<const int64_t *>(pos_ids->data()),
            reinterpret_cast<const int64_t *>(slot_ids->data()),
            reinterpret_cast<const int64_t *>(positions->data()),
            theta, q->dtype(), q->shape()[0], k_cache->shape()[1], q->shape()[1],
            k->shape()[1], q->shape()[2]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void split_qkv_decode(tensor_t q, tensor_t k, tensor_t v, tensor_t qkv) {
    CHECK_SAME_DEVICE(q, k, v, qkv);
    CHECK_SAME_DTYPE(q->dtype(), k->dtype(), v->dtype(), qkv->dtype());
    CHECK_ARGUMENT(q->ndim() == 2 && k->ndim() == 2 && v->ndim() == 2 && qkv->ndim() == 2,
                   "split_qkv_decode expects 2D tensors");
    CHECK_ARGUMENT(q->isContiguous() && k->isContiguous() && v->isContiguous() && qkv->isContiguous(),
                   "split_qkv_decode expects contiguous tensors");
    CHECK_ARGUMENT(q->shape()[0] == k->shape()[0] && q->shape()[0] == v->shape()[0] &&
                       q->shape()[0] == qkv->shape()[0],
                   "split_qkv_decode batch mismatch");
    CHECK_ARGUMENT(qkv->shape()[1] == q->shape()[1] + k->shape()[1] + v->shape()[1],
                   "split_qkv_decode total dim mismatch");

    llaisys::core::context().setDevice(qkv->deviceType(), qkv->deviceId());
    switch (qkv->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::split_qkv_decode(q->data(), k->data(), v->data(), qkv->data(), qkv->dtype(),
                                        qkv->shape()[0], q->shape()[1], k->shape()[1], v->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void split_gate_up_decode(tensor_t gate, tensor_t up, tensor_t gate_up) {
    CHECK_SAME_DEVICE(gate, up, gate_up);
    CHECK_SAME_DTYPE(gate->dtype(), up->dtype(), gate_up->dtype());
    CHECK_ARGUMENT(gate->ndim() == 2 && up->ndim() == 2 && gate_up->ndim() == 2,
                   "split_gate_up_decode expects 2D tensors");
    CHECK_ARGUMENT(gate->isContiguous() && up->isContiguous() && gate_up->isContiguous(),
                   "split_gate_up_decode expects contiguous tensors");
    CHECK_ARGUMENT(gate->shape() == up->shape(), "split_gate_up_decode gate/up shape mismatch");
    CHECK_ARGUMENT(gate_up->shape()[0] == gate->shape()[0] &&
                       gate_up->shape()[1] == gate->shape()[1] * 2,
                   "split_gate_up_decode total dim mismatch");

    llaisys::core::context().setDevice(gate_up->deviceType(), gate_up->deviceId());
    switch (gate_up->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::split_gate_up_decode(gate->data(), up->data(), gate_up->data(),
                                            gate_up->dtype(), gate_up->shape()[0], gate->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
