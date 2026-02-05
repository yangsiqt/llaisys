#ifndef LLAISYS_MODELS_QWEN2_HPP
#define LLAISYS_MODELS_QWEN2_HPP

#include "../../tensor/tensor.hpp"
#include <vector>
#include <memory>

namespace llaisys {
namespace models {

struct Qwen2Config {
    llaisysDataType_t dtype;
    size_t nlayer;      // num_hidden_layers
    size_t hs;          // hidden_size
    size_t nh;          // num_attention_heads
    size_t nkvh;        // num_key_value_heads
    size_t dh;          // head_dim = hidden_size / num_attention_heads
    size_t di;          // intermediate_size
    size_t maxseq;      // max_position_embeddings
    size_t voc;         // vocab_size
    float epsilon;      // rms_norm_eps
    float theta;        // rope_theta
    int64_t end_token;  // eos_token_id
};

struct Qwen2Weights {
    // Embedding layers
    tensor_t in_embed;     // token_embeddings
    tensor_t out_embed;    // lm_head
    tensor_t out_norm_w;   // model.norm.weight
    
    // Per-layer weights (size: nlayer)
    std::vector<tensor_t> attn_norm_w;  // input_layernorm.weight
    std::vector<tensor_t> attn_q_w;     // q_proj.weight
    std::vector<tensor_t> attn_q_b;     // q_proj.bias
    std::vector<tensor_t> attn_k_w;     // k_proj.weight
    std::vector<tensor_t> attn_k_b;     // k_proj.bias
    std::vector<tensor_t> attn_v_w;     // v_proj.weight
    std::vector<tensor_t> attn_v_b;     // v_proj.bias
    std::vector<tensor_t> attn_o_w;     // o_proj.weight
    std::vector<tensor_t> mlp_norm_w;   // post_attention_layernorm.weight
    std::vector<tensor_t> mlp_gate_w;   // gate_proj.weight
    std::vector<tensor_t> mlp_up_w;     // up_proj.weight
    std::vector<tensor_t> mlp_down_w;   // down_proj.weight
};

// KV Cache for one layer
struct KVCache {
    tensor_t k_cache;  // [max_seq, nkvh, dh]
    tensor_t v_cache;  // [max_seq, nkvh, dh]
    size_t current_seq_len;
    
    KVCache() : current_seq_len(0) {}
};

class Qwen2Model {
public:
    Qwen2Model(const Qwen2Config& config, llaisysDeviceType_t device_type, const std::vector<int>& device_ids);
    ~Qwen2Model();
    
    // Get weights for loading
    Qwen2Weights& weights() { return weights_; }
    const Qwen2Config& config() const { return config_; }
    
    // Inference: takes token_ids, returns next token id
    int64_t infer(const std::vector<int64_t>& token_ids);
    
    // Reset KV cache
    void reset_cache();
    
private:
    Qwen2Config config_;
    Qwen2Weights weights_;
    llaisysDeviceType_t device_type_;
    std::vector<int> device_ids_;
    
    // KV cache for all layers
    std::vector<KVCache> kv_caches_;
    size_t current_pos_;
    
    // Forward pass
    tensor_t forward(const tensor_t& input_ids, size_t start_pos);
    
    // Helper functions
    tensor_t apply_layer(size_t layer_idx, const tensor_t& hidden_states, size_t start_pos);
};

} // namespace models
} // namespace llaisys

#endif // LLAISYS_MODELS_QWEN2_HPP

