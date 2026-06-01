#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.hpp"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale,
                    size_t qlen_per_seq) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(q->ndim() == 3, "q must be 3D [qlen, n_heads, head_dim]");
    CHECK_ARGUMENT(k->ndim() == 3, "k must be 3D [kvlen, n_kv_heads, head_dim]");
    CHECK_ARGUMENT(v->ndim() == 3, "v must be 3D [kvlen, n_kv_heads, head_dim]");
    CHECK_ARGUMENT(attn_val->ndim() == 3, "attn_val must be 3D");
    CHECK_ARGUMENT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
                   "All tensors must be contiguous");

    size_t qlen = q->shape()[0];
    size_t kvlen = k->shape()[0];
    size_t n_heads = q->shape()[1];
    size_t n_kv_heads = k->shape()[1];
    size_t head_dim = q->shape()[2];

    if (qlen_per_seq == 0) qlen_per_seq = qlen;

    CHECK_ARGUMENT(qlen % qlen_per_seq == 0, "qlen must be divisible by qlen_per_seq");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0], "k and v must have same kvlen");
    CHECK_ARGUMENT(k->shape()[1] == v->shape()[1], "k and v must have same n_kv_heads");
    CHECK_ARGUMENT(k->shape()[2] == head_dim, "k must have same head_dim as q");
    CHECK_ARGUMENT(v->shape()[2] == head_dim, "v must have same head_dim as q");
    CHECK_ARGUMENT(n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads (GQA)");
    CHECK_ARGUMENT(attn_val->shape()[0] == qlen, "attn_val must have same qlen as q");
    CHECK_ARGUMENT(attn_val->shape()[1] == n_heads, "attn_val must have same n_heads as q");
    CHECK_ARGUMENT(attn_val->shape()[2] == head_dim, "attn_val must have same head_dim as q");

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  scale, attn_val->dtype(), qlen, kvlen, n_heads, n_kv_heads, head_dim,
                                  qlen_per_seq);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  scale, attn_val->dtype(), qlen, kvlen, n_heads, n_kv_heads, head_dim,
                                  qlen_per_seq);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
                                      qlen, kvlen, n_heads, n_kv_heads, head_dim, qlen_per_seq);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void self_attention_slots_decode(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache,
                                 tensor_t slot_ids, tensor_t seq_lens, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k_cache, v_cache, slot_ids, seq_lens);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k_cache->dtype(), v_cache->dtype());
    CHECK_ARGUMENT(q->ndim() == 3, "q must be 3D [batch, n_heads, head_dim]");
    CHECK_ARGUMENT(k_cache->ndim() == 4, "k_cache must be 4D [max_slots, maxseq, n_kv_heads, head_dim]");
    CHECK_ARGUMENT(v_cache->ndim() == 4, "v_cache must be 4D [max_slots, maxseq, n_kv_heads, head_dim]");
    CHECK_ARGUMENT(attn_val->ndim() == 3, "attn_val must be 3D");
    CHECK_ARGUMENT(slot_ids->ndim() == 1 && seq_lens->ndim() == 1, "slot_ids and seq_lens must be 1D");
    CHECK_ARGUMENT(slot_ids->dtype() == LLAISYS_DTYPE_I64, "slot_ids must be int64");
    CHECK_ARGUMENT(seq_lens->dtype() == LLAISYS_DTYPE_I64, "seq_lens must be int64");
    CHECK_ARGUMENT(attn_val->isContiguous() && q->isContiguous() && k_cache->isContiguous() &&
                       v_cache->isContiguous() && slot_ids->isContiguous() && seq_lens->isContiguous(),
                   "All tensors must be contiguous");

    size_t batch_size = q->shape()[0];
    size_t n_heads = q->shape()[1];
    size_t head_dim = q->shape()[2];
    size_t maxseq = k_cache->shape()[1];
    size_t n_kv_heads = k_cache->shape()[2];

    CHECK_ARGUMENT(k_cache->shape()[0] == v_cache->shape()[0], "K/V cache max_slots mismatch");
    CHECK_ARGUMENT(k_cache->shape()[1] == v_cache->shape()[1], "K/V cache maxseq mismatch");
    CHECK_ARGUMENT(k_cache->shape()[2] == v_cache->shape()[2], "K/V cache n_kv_heads mismatch");
    CHECK_ARGUMENT(k_cache->shape()[3] == head_dim && v_cache->shape()[3] == head_dim,
                   "K/V cache head_dim mismatch");
    CHECK_ARGUMENT(n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads (GQA)");
    CHECK_ARGUMENT(attn_val->shape()[0] == batch_size, "attn_val batch mismatch");
    CHECK_ARGUMENT(attn_val->shape()[1] == n_heads, "attn_val n_heads mismatch");
    CHECK_ARGUMENT(attn_val->shape()[2] == head_dim, "attn_val head_dim mismatch");
    CHECK_ARGUMENT(slot_ids->shape()[0] == batch_size && seq_lens->shape()[0] == batch_size,
                   "slot_ids/seq_lens batch mismatch");

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention_slots_decode(attn_val->data(), q->data(), k_cache->data(), v_cache->data(),
                                                   reinterpret_cast<const int64_t *>(slot_ids->data()),
                                                   reinterpret_cast<const int64_t *>(seq_lens->data()), scale,
                                                   attn_val->dtype(), batch_size, maxseq, n_heads, n_kv_heads,
                                                   head_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void self_attention_gqa_slots_decode(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache,
                                     tensor_t slot_ids, tensor_t seq_lens, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k_cache, v_cache, slot_ids, seq_lens);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k_cache->dtype(), v_cache->dtype());
    CHECK_ARGUMENT(q->ndim() == 3, "q must be 3D [batch, n_heads, head_dim]");
    CHECK_ARGUMENT(k_cache->ndim() == 4, "k_cache must be 4D [max_slots, maxseq, n_kv_heads, head_dim]");
    CHECK_ARGUMENT(v_cache->ndim() == 4, "v_cache must be 4D [max_slots, maxseq, n_kv_heads, head_dim]");
    CHECK_ARGUMENT(attn_val->ndim() == 3, "attn_val must be 3D");
    CHECK_ARGUMENT(slot_ids->dtype() == LLAISYS_DTYPE_I64, "slot_ids must be int64");
    CHECK_ARGUMENT(seq_lens->dtype() == LLAISYS_DTYPE_I64, "seq_lens must be int64");
    CHECK_ARGUMENT(attn_val->isContiguous() && q->isContiguous() && k_cache->isContiguous() &&
                       v_cache->isContiguous() && slot_ids->isContiguous() && seq_lens->isContiguous(),
                   "All tensors must be contiguous");

    size_t batch_size = q->shape()[0];
    size_t n_heads = q->shape()[1];
    size_t head_dim = q->shape()[2];
    size_t maxseq = k_cache->shape()[1];
    size_t n_kv_heads = k_cache->shape()[2];
    CHECK_ARGUMENT(k_cache->shape()[0] == v_cache->shape()[0], "K/V cache max_slots mismatch");
    CHECK_ARGUMENT(k_cache->shape()[1] == v_cache->shape()[1], "K/V cache maxseq mismatch");
    CHECK_ARGUMENT(k_cache->shape()[2] == v_cache->shape()[2], "K/V cache n_kv_heads mismatch");
    CHECK_ARGUMENT(k_cache->shape()[3] == head_dim && v_cache->shape()[3] == head_dim,
                   "K/V cache head_dim mismatch");
    CHECK_ARGUMENT(n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads");
    CHECK_ARGUMENT(attn_val->shape()[0] == batch_size && attn_val->shape()[1] == n_heads &&
                       attn_val->shape()[2] == head_dim,
                   "attn_val shape mismatch");
    CHECK_ARGUMENT(slot_ids->shape()[0] == batch_size && seq_lens->shape()[0] == batch_size,
                   "slot_ids/seq_lens batch mismatch");

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention_gqa_slots_decode(
            attn_val->data(), q->data(), k_cache->data(), v_cache->data(),
            reinterpret_cast<const int64_t *>(slot_ids->data()),
            reinterpret_cast<const int64_t *>(seq_lens->data()), scale, attn_val->dtype(), batch_size,
            maxseq, n_heads, n_kv_heads, head_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
