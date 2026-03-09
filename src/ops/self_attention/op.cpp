#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.hpp"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
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
                                  scale, attn_val->dtype(), qlen, kvlen, n_heads, n_kv_heads, head_dim);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), 
                                  scale, attn_val->dtype(), qlen, kvlen, n_heads, n_kv_heads, head_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), qlen, kvlen, n_heads, n_kv_heads, head_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
