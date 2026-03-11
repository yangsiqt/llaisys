#include "qwen2_tp.hpp"
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
    CHECK_ARGUMENT(dst->dtype() == src->dtype(), "dtype mismatch");
    CHECK_ARGUMENT(dst->isContiguous() && src->isContiguous(), "tensors must be contiguous");
    CHECK_ARGUMENT(dst->numel() == src->numel(), "numel mismatch");

    size_t bytes = src->numel() * src->elementSize();
    core::context().setDevice(dst->deviceType(), dst->deviceId());
    llaisysMemcpyKind_t kind =
        (dst->deviceType() == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D;
    core::context().runtime().api()->memcpy_sync(dst->data(), src->data(), bytes, kind);
}

Qwen2TPModel::Qwen2TPModel(const Qwen2Config& config, llaisysDeviceType_t device_type,
                            const std::vector<int>& device_ids)
    : config_(config), device_type_(device_type), device_ids_(device_ids),
      tp_size_(device_ids.size()), current_pos_(0) {

    CHECK_ARGUMENT(tp_size_ > 0, "need at least 1 device");
    CHECK_ARGUMENT(config_.nh % tp_size_ == 0, "num_attention_heads must be divisible by tp_size");
    CHECK_ARGUMENT(config_.nkvh % tp_size_ == 0, "num_key_value_heads must be divisible by tp_size");
    CHECK_ARGUMENT(config_.di % tp_size_ == 0, "intermediate_size must be divisible by tp_size");

    nh_per_rank_ = config_.nh / tp_size_;
    nkvh_per_rank_ = config_.nkvh / tp_size_;
    di_per_rank_ = config_.di / tp_size_;

    std::cout << "[TP] Tensor Parallel size: " << tp_size_ << std::endl;
    std::cout << "[TP] nh_per_rank=" << nh_per_rank_
              << " nkvh_per_rank=" << nkvh_per_rank_
              << " di_per_rank=" << di_per_rank_ << std::endl;

    ranks_.resize(tp_size_);
    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        auto& rd = ranks_[rank];
        rd.kv_caches.resize(config_.nlayer);

        for (size_t i = 0; i < config_.nlayer; i++) {
            core::context().setDevice(device_type_, dev_id);
            rd.kv_caches[i].k_cache = Tensor::create(
                {config_.maxseq, nkvh_per_rank_, config_.dh},
                config_.dtype, device_type_, dev_id);
            rd.kv_caches[i].v_cache = Tensor::create(
                {config_.maxseq, nkvh_per_rank_, config_.dh},
                config_.dtype, device_type_, dev_id);
            rd.kv_caches[i].current_seq_len = 0;
        }

        auto& w = rd.weights;
        w.attn_norm_w.resize(config_.nlayer);
        w.attn_q_w.resize(config_.nlayer);
        w.attn_q_b.resize(config_.nlayer);
        w.attn_k_w.resize(config_.nlayer);
        w.attn_k_b.resize(config_.nlayer);
        w.attn_v_w.resize(config_.nlayer);
        w.attn_v_b.resize(config_.nlayer);
        w.attn_o_w.resize(config_.nlayer);
        w.mlp_norm_w.resize(config_.nlayer);
        w.mlp_gate_w.resize(config_.nlayer);
        w.mlp_up_w.resize(config_.nlayer);
        w.mlp_down_w.resize(config_.nlayer);
    }

#ifdef ENABLE_NVIDIA_API
    if (device_type_ == LLAISYS_DEVICE_NVIDIA && tp_size_ > 1) {
        nccl_comm_ = std::make_unique<device::nvidia::NcclComm>(device_ids_);
    }
#endif
}

Qwen2TPModel::~Qwen2TPModel() = default;

void Qwen2TPModel::reset_cache() {
    current_pos_ = 0;
    for (auto& rd : ranks_) {
        for (auto& cache : rd.kv_caches) {
            cache.current_seq_len = 0;
        }
    }
}

void Qwen2TPModel::setInEmbed(int rank, const tensor_t& tensor) {
    CHECK_ARGUMENT(rank >= 0 && rank < tp_size_, "invalid rank");
    ranks_[rank].weights.in_embed = tensor;
}

void Qwen2TPModel::setOutEmbed(const tensor_t& tensor) {
    ranks_[0].weights.out_embed = tensor;
}

void Qwen2TPModel::setOutNormW(int rank, const tensor_t& tensor) {
    CHECK_ARGUMENT(rank >= 0 && rank < tp_size_, "invalid rank");
    ranks_[rank].weights.out_norm_w = tensor;
}

void Qwen2TPModel::setLayerWeight(int rank, const std::string& name, size_t layer_idx, const tensor_t& tensor) {
    CHECK_ARGUMENT(rank >= 0 && rank < tp_size_, "invalid rank");
    CHECK_ARGUMENT(layer_idx < config_.nlayer, "invalid layer_idx");
    auto& w = ranks_[rank].weights;

    if (name == "attn_norm_w") w.attn_norm_w[layer_idx] = tensor;
    else if (name == "attn_q_w") w.attn_q_w[layer_idx] = tensor;
    else if (name == "attn_q_b") w.attn_q_b[layer_idx] = tensor;
    else if (name == "attn_k_w") w.attn_k_w[layer_idx] = tensor;
    else if (name == "attn_k_b") w.attn_k_b[layer_idx] = tensor;
    else if (name == "attn_v_w") w.attn_v_w[layer_idx] = tensor;
    else if (name == "attn_v_b") w.attn_v_b[layer_idx] = tensor;
    else if (name == "attn_o_w") w.attn_o_w[layer_idx] = tensor;
    else if (name == "mlp_norm_w") w.mlp_norm_w[layer_idx] = tensor;
    else if (name == "mlp_gate_w") w.mlp_gate_w[layer_idx] = tensor;
    else if (name == "mlp_up_w") w.mlp_up_w[layer_idx] = tensor;
    else if (name == "mlp_down_w") w.mlp_down_w[layer_idx] = tensor;
}

void Qwen2TPModel::allreduce_sum(std::vector<tensor_t>& tensors) {
    if (tp_size_ <= 1) return;

#ifdef ENABLE_NVIDIA_API
    if (device_type_ == LLAISYS_DEVICE_NVIDIA && nccl_comm_) {
        std::vector<void*> bufs(tp_size_);
        for (int i = 0; i < tp_size_; i++) {
            bufs[i] = tensors[i]->data();
        }
        nccl_comm_->allreduceSum(bufs, tensors[0]->numel(), tensors[0]->dtype());
    }
#endif
}

int64_t Qwen2TPModel::infer(const std::vector<int64_t>& token_ids) {
    CHECK_ARGUMENT(!token_ids.empty(), "token_ids must not be empty");

    if (token_ids.size() < current_pos_) {
        reset_cache();
    }

    size_t start_pos = current_pos_;
    size_t total_len = token_ids.size();
    size_t seq_len = total_len - start_pos;
    CHECK_ARGUMENT(seq_len > 0, "no new tokens to process");
    CHECK_ARGUMENT(total_len <= config_.maxseq, "sequence length exceeds maxseq");

    std::vector<int64_t> new_tokens(token_ids.begin() + start_pos, token_ids.end());

    tensor_t logits = forward(new_tokens, start_pos, seq_len);

    current_pos_ = total_len;

    int dev_id = device_ids_[0];
    core::context().setDevice(device_type_, dev_id);

    tensor_t last_logits = logits->slice(0, seq_len - 1, seq_len);
    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, dev_id);
    tensor_t max_val = Tensor::create({1}, config_.dtype, device_type_, dev_id);
    ops::argmax(max_idx, max_val, last_logits->view({config_.voc}));

    int64_t next_token;
    std::vector<std::byte> buffer(sizeof(int64_t));
    core::context().runtime().api()->memcpy_sync(
        buffer.data(), max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
    std::memcpy(&next_token, buffer.data(), sizeof(int64_t));

    return next_token;
}

tensor_t Qwen2TPModel::forward(const std::vector<int64_t>& new_tokens, size_t start_pos, size_t seq_len) {
    std::vector<tensor_t> hidden_states(tp_size_);

    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        core::context().setDevice(device_type_, dev_id);

        tensor_t input_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        input_ids->load(new_tokens.data());

        hidden_states[rank] = Tensor::create({seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
        ops::embedding(hidden_states[rank], input_ids, ranks_[rank].weights.in_embed);
    }

    for (size_t layer = 0; layer < config_.nlayer; layer++) {
        apply_layer(layer, hidden_states, start_pos, seq_len);
    }

    int dev_id = device_ids_[0];
    core::context().setDevice(device_type_, dev_id);

    tensor_t normed = Tensor::create({seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
    ops::rms_norm(normed, hidden_states[0], ranks_[0].weights.out_norm_w, config_.epsilon);

    tensor_t logits = Tensor::create({seq_len, config_.voc}, config_.dtype, device_type_, dev_id);
    ops::linear(logits, normed, ranks_[0].weights.out_embed, nullptr);

    return logits;
}

void Qwen2TPModel::apply_layer(size_t layer_idx, std::vector<tensor_t>& hidden_states,
                                size_t start_pos, size_t seq_len) {
    size_t kv_seq_len = start_pos + seq_len;
    float scale = 1.0f / std::sqrt(static_cast<float>(config_.dh));

    std::vector<tensor_t> o_proj(tp_size_);
    std::vector<tensor_t> hidden_states_1(tp_size_);
    std::vector<tensor_t> mlp_out(tp_size_);

    // Phase 1: Attention (each rank independently)
    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        core::context().setDevice(device_type_, dev_id);
        auto& w = ranks_[rank].weights;

        // Attention norm
        tensor_t attn_norm_out = Tensor::create(
            {seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
        ops::rms_norm(attn_norm_out, hidden_states[rank], w.attn_norm_w[layer_idx], config_.epsilon);

        // Q/K/V projections (column-parallel: sharded output dim)
        tensor_t q_flat = Tensor::create(
            {seq_len, nh_per_rank_ * config_.dh}, config_.dtype, device_type_, dev_id);
        tensor_t k_flat = Tensor::create(
            {seq_len, nkvh_per_rank_ * config_.dh}, config_.dtype, device_type_, dev_id);
        tensor_t v_flat = Tensor::create(
            {seq_len, nkvh_per_rank_ * config_.dh}, config_.dtype, device_type_, dev_id);

        ops::linear(q_flat, attn_norm_out, w.attn_q_w[layer_idx], w.attn_q_b[layer_idx]);
        ops::linear(k_flat, attn_norm_out, w.attn_k_w[layer_idx], w.attn_k_b[layer_idx]);
        ops::linear(v_flat, attn_norm_out, w.attn_v_w[layer_idx], w.attn_v_b[layer_idx]);

        tensor_t q = q_flat->view({seq_len, nh_per_rank_, config_.dh});
        tensor_t k = k_flat->view({seq_len, nkvh_per_rank_, config_.dh});
        tensor_t v = v_flat->view({seq_len, nkvh_per_rank_, config_.dh});

        // RoPE
        std::vector<int64_t> pos_ids_vec(seq_len);
        for (size_t i = 0; i < seq_len; i++) pos_ids_vec[i] = start_pos + i;
        tensor_t pos_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        pos_ids->load(pos_ids_vec.data());
        ops::rope(q, q, pos_ids, config_.theta);
        ops::rope(k, k, pos_ids, config_.theta);

        // KV cache update
        CHECK_ARGUMENT(kv_seq_len <= config_.maxseq, "kv_seq_len exceeds maxseq");
        auto& kv_cache = ranks_[rank].kv_caches[layer_idx];
        tensor_t k_dst = kv_cache.k_cache->slice(0, start_pos, kv_seq_len);
        tensor_t v_dst = kv_cache.v_cache->slice(0, start_pos, kv_seq_len);
        copy_contiguous_tensor_(k_dst, k);
        copy_contiguous_tensor_(v_dst, v);

        tensor_t full_k = kv_cache.k_cache->slice(0, 0, kv_seq_len);
        tensor_t full_v = kv_cache.v_cache->slice(0, 0, kv_seq_len);
        kv_cache.current_seq_len = kv_seq_len;

        // Self-attention
        tensor_t attn_out = Tensor::create(
            {seq_len, nh_per_rank_, config_.dh}, config_.dtype, device_type_, dev_id);
        ops::self_attention(attn_out, q, full_k, full_v, scale);

        // O projection (row-parallel: sharded input dim)
        tensor_t attn_out_flat = attn_out->view({seq_len, nh_per_rank_ * config_.dh});
        o_proj[rank] = Tensor::create({seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
        ops::linear(o_proj[rank], attn_out_flat, w.attn_o_w[layer_idx], nullptr);
    }

    // All-reduce after O projection
    allreduce_sum(o_proj);

    // Phase 2: Residual + MLP
    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        core::context().setDevice(device_type_, dev_id);
        auto& w = ranks_[rank].weights;

        // Residual
        hidden_states_1[rank] = Tensor::create(
            {seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
        ops::add(hidden_states_1[rank], hidden_states[rank], o_proj[rank]);

        // MLP norm
        tensor_t mlp_norm_out = Tensor::create(
            {seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
        ops::rms_norm(mlp_norm_out, hidden_states_1[rank], w.mlp_norm_w[layer_idx], config_.epsilon);

        // Gate/Up projections (column-parallel)
        tensor_t gate_out = Tensor::create(
            {seq_len, di_per_rank_}, config_.dtype, device_type_, dev_id);
        tensor_t up_out = Tensor::create(
            {seq_len, di_per_rank_}, config_.dtype, device_type_, dev_id);
        ops::linear(gate_out, mlp_norm_out, w.mlp_gate_w[layer_idx], nullptr);
        ops::linear(up_out, mlp_norm_out, w.mlp_up_w[layer_idx], nullptr);

        // SwiGLU
        tensor_t swiglu_out = Tensor::create(
            {seq_len, di_per_rank_}, config_.dtype, device_type_, dev_id);
        ops::swiglu(swiglu_out, gate_out, up_out);

        // Down projection (row-parallel)
        mlp_out[rank] = Tensor::create(
            {seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
        ops::linear(mlp_out[rank], swiglu_out, w.mlp_down_w[layer_idx], nullptr);
    }

    // All-reduce after Down projection
    allreduce_sum(mlp_out);

    // Phase 3: Final residual
    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        core::context().setDevice(device_type_, dev_id);

        tensor_t output = Tensor::create(
            {seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
        ops::add(output, hidden_states_1[rank], mlp_out[rank]);
        hidden_states[rank] = output;
    }
}

} // namespace models
} // namespace llaisys
