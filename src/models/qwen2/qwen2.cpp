#include "qwen2.hpp"
#include "../../ops/ops.hpp"
#include "../../utils.hpp"
#include "../../core/context/context.hpp"
#include <iostream>
#include <cstring>
#include <cmath>

namespace llaisys {
namespace models {

static inline void copy_contiguous_tensor_(const tensor_t &dst, const tensor_t &src) {
    CHECK_ARGUMENT(dst && src, "copy_contiguous_tensor_: null tensor");
    CHECK_SAME_DEVICE(dst, src);
    CHECK_ARGUMENT(dst->dtype() == src->dtype(), "copy_contiguous_tensor_: dtype mismatch");
    CHECK_ARGUMENT(dst->isContiguous() && src->isContiguous(), "copy_contiguous_tensor_: tensors must be contiguous");
    CHECK_ARGUMENT(dst->numel() == src->numel(), "copy_contiguous_tensor_: numel mismatch");

    size_t bytes = src->numel() * src->elementSize();
    core::context().setDevice(dst->deviceType(), dst->deviceId());
    llaisysMemcpyKind_t kind =
        (dst->deviceType() == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D;
    core::context().runtime().api()->memcpy_sync(dst->data(), src->data(), bytes, kind);
}

Qwen2Model::Qwen2Model(const Qwen2Config& config, llaisysDeviceType_t device_type, const std::vector<int>& device_ids)
    : config_(config), device_type_(device_type), device_ids_(device_ids), current_pos_(0) {
    
    // Initialize KV caches for all layers
    kv_caches_.resize(config_.nlayer);
    
    int device_id = device_ids_.empty() ? 0 : device_ids_[0];
    
    // Create KV cache tensors for each layer
    for (size_t i = 0; i < config_.nlayer; i++) {
        // k_cache: [max_seq, nkvh, dh]
        kv_caches_[i].k_cache = Tensor::create(
            {config_.maxseq, config_.nkvh, config_.dh},
            config_.dtype,
            device_type_,
            device_id
        );
        
        // v_cache: [max_seq, nkvh, dh]
        kv_caches_[i].v_cache = Tensor::create(
            {config_.maxseq, config_.nkvh, config_.dh},
            config_.dtype,
            device_type_,
            device_id
        );
        
        kv_caches_[i].current_seq_len = 0;
    }
}

Qwen2Model::~Qwen2Model() {
    // Tensors are managed by shared_ptr, will be cleaned up automatically
}

void Qwen2Model::reset_cache() {
    current_pos_ = 0;
    for (auto& cache : kv_caches_) {
        cache.current_seq_len = 0;
    }
}

int64_t Qwen2Model::infer(const std::vector<int64_t>& token_ids) {
    CHECK_ARGUMENT(!token_ids.empty(), "infer: token_ids must not be empty");

    // Heuristic: if a new (shorter) sequence is provided, reset KV cache.
    if (token_ids.size() < current_pos_) {
        reset_cache();
    }

    // Only process newly appended tokens (incremental inference with KV cache).
    size_t start_pos = current_pos_;
    size_t total_len = token_ids.size();
    size_t seq_len = total_len - start_pos;
    CHECK_ARGUMENT(seq_len > 0, "infer: no new tokens to process");
    CHECK_ARGUMENT(total_len <= config_.maxseq, "infer: sequence length exceeds maxseq");

    // Create input tensor for new tokens only.
    int device_id = device_ids_.empty() ? 0 : device_ids_[0];
    tensor_t input_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_type_, device_id);

    input_ids->load(token_ids.data() + start_pos);

    // Forward pass for the new segment only (will attend to cached K/V).
    tensor_t logits = forward(input_ids, start_pos);

    // Update position to total cached length.
    current_pos_ = total_len;
    
    // Get the last token's logits and find argmax
    // logits shape: [seq_len, vocab_size]
    tensor_t last_logits = logits->slice(0, seq_len - 1, seq_len);  // [1, vocab_size]
    
    // Argmax to get next token
    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, device_id);
    tensor_t max_val = Tensor::create({1}, config_.dtype, device_type_, device_id);
    ops::argmax(max_idx, max_val, last_logits->view({config_.voc}));
    
    // Get result from tensor
    int64_t next_token;
    std::vector<std::byte> buffer(sizeof(int64_t));
    core::context().setDevice(device_type_, device_id);
    core::context().runtime().api()->memcpy_sync(
        buffer.data(), max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H
    );
    std::memcpy(&next_token, buffer.data(), sizeof(int64_t));
    
    return next_token;
}

tensor_t Qwen2Model::forward(const tensor_t& input_ids, size_t start_pos) {
    size_t seq_len = input_ids->shape()[0];
    int device_id = device_ids_.empty() ? 0 : device_ids_[0];
    
    // Token embedding: [seq_len] -> [seq_len, hidden_size]
    tensor_t hidden_states = Tensor::create({seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    
    ops::embedding(hidden_states, input_ids, weights_.in_embed);
    
    // Apply each transformer layer
    for (size_t layer_idx = 0; layer_idx < config_.nlayer; layer_idx++) {
        hidden_states = apply_layer(layer_idx, hidden_states, start_pos);
    }
    
    // Final layer norm
    tensor_t normed = Tensor::create({seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::rms_norm(normed, hidden_states, weights_.out_norm_w, config_.epsilon);
    
    // Output projection: [seq_len, hidden_size] -> [seq_len, vocab_size]
    tensor_t logits = Tensor::create({seq_len, config_.voc}, config_.dtype, device_type_, device_id);
    ops::linear(logits, normed, weights_.out_embed, nullptr);
    
    return logits;
}

tensor_t Qwen2Model::apply_layer(size_t layer_idx, const tensor_t& hidden_states, size_t start_pos) {
    size_t seq_len = hidden_states->shape()[0];
    int device_id = device_ids_.empty() ? 0 : device_ids_[0];
    
    // 1. Attention norm
    tensor_t attn_norm_out = Tensor::create({seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::rms_norm(attn_norm_out, hidden_states, weights_.attn_norm_w[layer_idx], config_.epsilon);
    
    // 2. Q, K, V projections
    tensor_t q = Tensor::create({seq_len, config_.nh, config_.dh}, config_.dtype, device_type_, device_id);
    tensor_t k = Tensor::create({seq_len, config_.nkvh, config_.dh}, config_.dtype, device_type_, device_id);
    tensor_t v = Tensor::create({seq_len, config_.nkvh, config_.dh}, config_.dtype, device_type_, device_id);
    
    // Reshape attn_norm_out for linear projection
    tensor_t q_flat = Tensor::create({seq_len, config_.nh * config_.dh}, config_.dtype, device_type_, device_id);
    tensor_t k_flat = Tensor::create({seq_len, config_.nkvh * config_.dh}, config_.dtype, device_type_, device_id);
    tensor_t v_flat = Tensor::create({seq_len, config_.nkvh * config_.dh}, config_.dtype, device_type_, device_id);
    
    ops::linear(q_flat, attn_norm_out, weights_.attn_q_w[layer_idx], weights_.attn_q_b[layer_idx]);
    ops::linear(k_flat, attn_norm_out, weights_.attn_k_w[layer_idx], weights_.attn_k_b[layer_idx]);
    ops::linear(v_flat, attn_norm_out, weights_.attn_v_w[layer_idx], weights_.attn_v_b[layer_idx]);
    
    // Reshape to [seq_len, num_heads, head_dim]
    q = q_flat->view({seq_len, config_.nh, config_.dh});
    k = k_flat->view({seq_len, config_.nkvh, config_.dh});
    v = v_flat->view({seq_len, config_.nkvh, config_.dh});
    
    // 3. Apply RoPE
    // Create position IDs tensor
    std::vector<int64_t> pos_ids_vec(seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        pos_ids_vec[i] = start_pos + i;
    }
    tensor_t pos_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_type_, device_id);
    pos_ids->load(pos_ids_vec.data());
    
    // Apply RoPE in-place
    ops::rope(q, q, pos_ids, config_.theta);
    ops::rope(k, k, pos_ids, config_.theta);
    
    // 4. Update KV cache and get full K, V
    tensor_t full_k, full_v;
    size_t kv_seq_len = start_pos + seq_len;

    // Copy the newly computed K/V (for this segment) into KV cache at [start_pos : kv_seq_len).
    CHECK_ARGUMENT(kv_seq_len <= config_.maxseq, "apply_layer: kv_seq_len exceeds maxseq");
    tensor_t k_dst = kv_caches_[layer_idx].k_cache->slice(0, start_pos, kv_seq_len);
    tensor_t v_dst = kv_caches_[layer_idx].v_cache->slice(0, start_pos, kv_seq_len);
    copy_contiguous_tensor_(k_dst, k);
    copy_contiguous_tensor_(v_dst, v);

    // Use full cached K/V for attention.
    full_k = kv_caches_[layer_idx].k_cache->slice(0, 0, kv_seq_len);
    full_v = kv_caches_[layer_idx].v_cache->slice(0, 0, kv_seq_len);

    kv_caches_[layer_idx].current_seq_len = kv_seq_len;
    
    // 5. Self-attention
    // Q: [seq_len, nh, dh], K,V: [kv_seq_len, nkvh, dh]
    tensor_t attn_out = Tensor::create({seq_len, config_.nh, config_.dh}, config_.dtype, device_type_, device_id);
    float scale = 1.0f / std::sqrt(static_cast<float>(config_.dh));
    ops::self_attention(attn_out, q, full_k, full_v, scale);
    
    // 6. Output projection
    tensor_t attn_out_flat = attn_out->view({seq_len, config_.nh * config_.dh});
    tensor_t o_proj = Tensor::create({seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::linear(o_proj, attn_out_flat, weights_.attn_o_w[layer_idx], nullptr);
    
    // 7. Residual connection
    tensor_t hidden_states_1 = Tensor::create({seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::add(hidden_states_1, hidden_states, o_proj);
    
    // 8. MLP norm
    tensor_t mlp_norm_out = Tensor::create({seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::rms_norm(mlp_norm_out, hidden_states_1, weights_.mlp_norm_w[layer_idx], config_.epsilon);
    
    // 9. MLP layers
    tensor_t gate_out = Tensor::create({seq_len, config_.di}, config_.dtype, device_type_, device_id);
    tensor_t up_out = Tensor::create({seq_len, config_.di}, config_.dtype, device_type_, device_id);
    
    ops::linear(gate_out, mlp_norm_out, weights_.mlp_gate_w[layer_idx], nullptr);
    ops::linear(up_out, mlp_norm_out, weights_.mlp_up_w[layer_idx], nullptr);
    
    // 10. SwiGLU activation
    tensor_t swiglu_out = Tensor::create({seq_len, config_.di}, config_.dtype, device_type_, device_id);
    ops::swiglu(swiglu_out, gate_out, up_out);
    
    // 11. Down projection
    tensor_t mlp_out = Tensor::create({seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::linear(mlp_out, swiglu_out, weights_.mlp_down_w[layer_idx], nullptr);
    
    // 12. Residual connection
    tensor_t output = Tensor::create({seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::add(output, hidden_states_1, mlp_out);
    
    return output;
}

} // namespace models
} // namespace llaisys

