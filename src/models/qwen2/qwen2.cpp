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
    : config_(config), device_type_(device_type), device_ids_(device_ids), current_pos_(0), batch_size_(1) {

    // Initialize KV caches for all layers
    kv_caches_.resize(config_.nlayer);

    int device_id = device_ids_.empty() ? 0 : device_ids_[0];

    // Create KV cache tensors for each layer
    // Shape: [batch_size, maxseq, nkvh, dh] — batch_size will be set before use
    for (size_t i = 0; i < config_.nlayer; i++) {
        kv_caches_[i].k_cache = Tensor::create(
            {batch_size_, config_.maxseq, config_.nkvh, config_.dh},
            config_.dtype,
            device_type_,
            device_id
        );

        kv_caches_[i].v_cache = Tensor::create(
            {batch_size_, config_.maxseq, config_.nkvh, config_.dh},
            config_.dtype,
            device_type_,
            device_id
        );

        kv_caches_[i].current_seq_len = 0;
    }
}

Qwen2Model::~Qwen2Model() = default;

void Qwen2Model::reset_cache() {
    current_pos_ = 0;
    for (auto& cache : kv_caches_) {
        cache.current_seq_len = 0;
    }
}

int64_t Qwen2Model::infer(const std::vector<int64_t>& token_ids) {
    auto result = infer_batch(token_ids, 1);
    return result[0];
}

std::vector<int64_t> Qwen2Model::infer_batch(const std::vector<int64_t>& token_ids, size_t batch_size) {
    CHECK_ARGUMENT(!token_ids.empty(), "infer_batch: token_ids must not be empty");
    CHECK_ARGUMENT(batch_size >= 1, "infer_batch: batch_size must be >= 1");
    CHECK_ARGUMENT(token_ids.size() % batch_size == 0, "infer_batch: token_ids size must be divisible by batch_size");

    size_t total_len = token_ids.size() / batch_size;

    if (total_len < current_pos_) {
        reset_cache();
    }

    size_t start_pos = current_pos_;
    size_t seq_len = total_len - start_pos;
    CHECK_ARGUMENT(seq_len > 0, "infer_batch: no new tokens to process");
    CHECK_ARGUMENT(total_len <= config_.maxseq, "infer_batch: sequence length exceeds maxseq");

    // Recreate KV caches if batch_size changed
    if (batch_size != batch_size_) {
        int device_id = device_ids_.empty() ? 0 : device_ids_[0];
        batch_size_ = batch_size;
        for (size_t i = 0; i < config_.nlayer; i++) {
            kv_caches_[i].k_cache = Tensor::create(
                {batch_size_, config_.maxseq, config_.nkvh, config_.dh},
                config_.dtype, device_type_, device_id);
            kv_caches_[i].v_cache = Tensor::create(
                {batch_size_, config_.maxseq, config_.nkvh, config_.dh},
                config_.dtype, device_type_, device_id);
            kv_caches_[i].current_seq_len = 0;
        }
        current_pos_ = 0;
        start_pos = 0;
    }

    // Build input_ids: [batch_size * seq_len], each sequence's new tokens
    std::vector<int64_t> new_tokens_vec;
    new_tokens_vec.reserve(batch_size * seq_len);
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < seq_len; i++) {
            new_tokens_vec.push_back(token_ids[b * total_len + start_pos + i]);
        }
    }

    int device_id = device_ids_.empty() ? 0 : device_ids_[0];
    core::context().setDevice(device_type_, device_id);

    tensor_t input_ids = Tensor::create({batch_size * seq_len}, LLAISYS_DTYPE_I64, device_type_, device_id);
    input_ids->load(new_tokens_vec.data());

    tensor_t logits = forward(input_ids, start_pos, batch_size);

    current_pos_ = total_len;

    // logits shape: [batch_size * seq_len, voc]
    // Extract last token's logit for each sequence
    std::vector<int64_t> next_tokens(batch_size);
    for (size_t b = 0; b < batch_size; b++) {
        size_t last_idx = (b + 1) * seq_len - 1;
        tensor_t last_logits = logits->slice(0, last_idx, last_idx + 1);

        tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, device_id);
        tensor_t max_val = Tensor::create({1}, config_.dtype, device_type_, device_id);
        ops::argmax(max_idx, max_val, last_logits->view({config_.voc}));

        std::vector<std::byte> buffer(sizeof(int64_t));
        core::context().setDevice(device_type_, device_id);
        core::context().runtime().api()->memcpy_sync(
            buffer.data(), max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
        std::memcpy(&next_tokens[b], buffer.data(), sizeof(int64_t));
    }

    return next_tokens;
}

tensor_t Qwen2Model::forward(const tensor_t& input_ids, size_t start_pos, size_t batch_size) {
    size_t seq_len = input_ids->shape()[0] / batch_size;
    int device_id = device_ids_.empty() ? 0 : device_ids_[0];

    // Token embedding: [batch_size * seq_len] -> [batch_size * seq_len, hidden_size]
    tensor_t hidden_states = Tensor::create({batch_size * seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::embedding(hidden_states, input_ids, weights_.in_embed);

    for (size_t layer_idx = 0; layer_idx < config_.nlayer; layer_idx++) {
        hidden_states = apply_layer(layer_idx, hidden_states, start_pos, batch_size);
    }

    // Final layer norm
    tensor_t normed = Tensor::create({batch_size * seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::rms_norm(normed, hidden_states, weights_.out_norm_w, config_.epsilon);

    // Output projection: [batch_size * seq_len, hidden_size] -> [batch_size * seq_len, vocab_size]
    tensor_t logits = Tensor::create({batch_size * seq_len, config_.voc}, config_.dtype, device_type_, device_id);
    ops::linear(logits, normed, weights_.out_embed, nullptr);

    return logits;
}

// Gather K/V from [batch_size, maxseq, nkvh, dh] cache into contiguous [batch_size*kv_seq_len, nkvh, dh].
// Each sequence's valid range is [0, kv_seq_len].
static tensor_t gather_kv_from_cache(const tensor_t& cache, size_t kv_seq_len,
                                      llaisysDataType_t dtype, llaisysDeviceType_t device_type, int device_id) {
    size_t batch_size = cache->shape()[0];

    auto flat = Tensor::create({batch_size * kv_seq_len, cache->shape()[2], cache->shape()[3]},
                               dtype, device_type, device_id);

    for (size_t b = 0; b < batch_size; b++) {
        auto seq_view = cache->slice(0, b, b + 1)            // [1, maxseq, nkvh, dh]
                            ->slice(1, 0, kv_seq_len)        // [1, kv_seq_len, nkvh, dh]
                            ->contiguous()                    // copy to contiguous
                            ->view({kv_seq_len, cache->shape()[2], cache->shape()[3]}); // 3D
        auto dst_view = flat->slice(0, b * kv_seq_len, (b + 1) * kv_seq_len);
        copy_contiguous_tensor_(dst_view, seq_view);
    }
    return flat;
}

// Scatter K/V from [batch_size*seq_len, nkvh, dh] into cache at positions [start_pos, start_pos+seq_len].
static void scatter_kv_to_cache(const tensor_t& cache, const tensor_t& kv_flat,
                                 size_t start_pos, size_t seq_len) {
    size_t batch_size = cache->shape()[0];
    auto kv_view = kv_flat->view({batch_size, seq_len, cache->shape()[2], cache->shape()[3]});

    for (size_t b = 0; b < batch_size; b++) {
        auto src = kv_view->slice(0, b, b + 1)          // [1, seq_len, nkvh, dh]
                       ->contiguous()
                       ->view({seq_len, cache->shape()[2], cache->shape()[3]});
        auto dst = cache->slice(0, b, b + 1)
                       ->slice(1, start_pos, start_pos + seq_len)
                       ->contiguous()
                       ->view({seq_len, cache->shape()[2], cache->shape()[3]});
        copy_contiguous_tensor_(dst, src);
    }
}

tensor_t Qwen2Model::apply_layer(size_t layer_idx, const tensor_t& hidden_states, size_t start_pos, size_t batch_size) {
    size_t flat_seq_len = hidden_states->shape()[0];
    size_t seq_len = flat_seq_len / batch_size;
    int device_id = device_ids_.empty() ? 0 : device_ids_[0];
    size_t kv_seq_len = start_pos + seq_len;

    // 1. Attention norm
    tensor_t attn_norm_out = Tensor::create({flat_seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::rms_norm(attn_norm_out, hidden_states, weights_.attn_norm_w[layer_idx], config_.epsilon);

    // 2. Q, K, V projections
    tensor_t q_flat = Tensor::create({flat_seq_len, config_.nh * config_.dh}, config_.dtype, device_type_, device_id);
    tensor_t k_flat = Tensor::create({flat_seq_len, config_.nkvh * config_.dh}, config_.dtype, device_type_, device_id);
    tensor_t v_flat = Tensor::create({flat_seq_len, config_.nkvh * config_.dh}, config_.dtype, device_type_, device_id);

    ops::linear(q_flat, attn_norm_out, weights_.attn_q_w[layer_idx], weights_.attn_q_b[layer_idx]);
    ops::linear(k_flat, attn_norm_out, weights_.attn_k_w[layer_idx], weights_.attn_k_b[layer_idx]);
    ops::linear(v_flat, attn_norm_out, weights_.attn_v_w[layer_idx], weights_.attn_v_b[layer_idx]);

    tensor_t q = q_flat->view({flat_seq_len, config_.nh, config_.dh});
    tensor_t k = k_flat->view({flat_seq_len, config_.nkvh, config_.dh});
    tensor_t v = v_flat->view({flat_seq_len, config_.nkvh, config_.dh});

    // 3. RoPE — position IDs within each sequence
    std::vector<int64_t> pos_ids_vec(flat_seq_len);
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < seq_len; i++) {
            pos_ids_vec[b * seq_len + i] = start_pos + i;
        }
    }
    tensor_t pos_ids = Tensor::create({flat_seq_len}, LLAISYS_DTYPE_I64, device_type_, device_id);
    pos_ids->load(pos_ids_vec.data());

    ops::rope(q, q, pos_ids, config_.theta);
    ops::rope(k, k, pos_ids, config_.theta);

    // 4. Update KV cache (scatter)
    CHECK_ARGUMENT(kv_seq_len <= config_.maxseq, "apply_layer: kv_seq_len exceeds maxseq");
    auto& kv_cache = kv_caches_[layer_idx];
    scatter_kv_to_cache(kv_cache.k_cache, k, start_pos, seq_len);
    scatter_kv_to_cache(kv_cache.v_cache, v, start_pos, seq_len);
    kv_cache.current_seq_len = kv_seq_len;

    // 5. Gather full K/V from cache for attention
    tensor_t full_k = gather_kv_from_cache(kv_cache.k_cache, kv_seq_len,
                                           config_.dtype, device_type_, device_id);
    tensor_t full_v = gather_kv_from_cache(kv_cache.v_cache, kv_seq_len,
                                           config_.dtype, device_type_, device_id);

    // 6. Self-attention
    float scale = 1.0f / std::sqrt(static_cast<float>(config_.dh));
    tensor_t attn_out = Tensor::create({flat_seq_len, config_.nh, config_.dh}, config_.dtype, device_type_, device_id);
    ops::self_attention(attn_out, q, full_k, full_v, scale, seq_len);

    // 7. Output projection
    tensor_t attn_out_flat = attn_out->view({flat_seq_len, config_.nh * config_.dh});
    tensor_t o_proj = Tensor::create({flat_seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::linear(o_proj, attn_out_flat, weights_.attn_o_w[layer_idx], nullptr);

    // 8. Residual connection
    tensor_t hidden_states_1 = Tensor::create({flat_seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::add(hidden_states_1, hidden_states, o_proj);

    // 9. MLP norm
    tensor_t mlp_norm_out = Tensor::create({flat_seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::rms_norm(mlp_norm_out, hidden_states_1, weights_.mlp_norm_w[layer_idx], config_.epsilon);

    // 10. MLP layers
    tensor_t gate_out = Tensor::create({flat_seq_len, config_.di}, config_.dtype, device_type_, device_id);
    tensor_t up_out = Tensor::create({flat_seq_len, config_.di}, config_.dtype, device_type_, device_id);

    ops::linear(gate_out, mlp_norm_out, weights_.mlp_gate_w[layer_idx], nullptr);
    ops::linear(up_out, mlp_norm_out, weights_.mlp_up_w[layer_idx], nullptr);

    // 11. SwiGLU activation
    tensor_t swiglu_out = Tensor::create({flat_seq_len, config_.di}, config_.dtype, device_type_, device_id);
    ops::swiglu(swiglu_out, gate_out, up_out);

    // 12. Down projection
    tensor_t mlp_out = Tensor::create({flat_seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::linear(mlp_out, swiglu_out, weights_.mlp_down_w[layer_idx], nullptr);

    // 13. Residual connection
    tensor_t output = Tensor::create({flat_seq_len, config_.hs}, config_.dtype, device_type_, device_id);
    ops::add(output, hidden_states_1, mlp_out);

    return output;
}

} // namespace models
} // namespace llaisys
