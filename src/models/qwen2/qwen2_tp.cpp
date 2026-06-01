#include "qwen2_tp.hpp"
#include "../../ops/ops.hpp"
#include "../../utils.hpp"
#include "../../core/context/context.hpp"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <chrono>

#ifdef ENABLE_NVIDIA_API
#include <cuda_runtime.h>
#endif

namespace llaisys {
namespace models {

static inline bool env_flag_enabled(const char* name) {
    const char* v = std::getenv(name);
    return v && std::atoi(v) != 0;
}

static inline double ms_since(std::chrono::steady_clock::time_point start) {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static constexpr size_t kDecodeArgmaxParts = 256;

static inline bool is_decode_graph_bucket(size_t batch_size) {
    return batch_size == 8 || batch_size == 16 || batch_size == 32 || batch_size == 64;
}

static inline void sync_device_for_timing(llaisysDeviceType_t device_type) {
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA) {
        cudaDeviceSynchronize();
    }
#else
    (void)device_type;
#endif
}

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

// Gather K/V from selected slots in [nslot, maxseq, nkvh, dh] cache into
// contiguous [slot_ids.size()*kv_seq_len, nkvh, dh].
static tensor_t gather_kv_from_cache_slots(const tensor_t& cache, const std::vector<size_t>& slot_ids,
                                           size_t kv_seq_len, llaisysDataType_t dtype,
                                           llaisysDeviceType_t device_type, int device_id) {
    size_t batch_size = slot_ids.size();

    auto flat = Tensor::create({batch_size * kv_seq_len, cache->shape()[2], cache->shape()[3]},
                               dtype, device_type, device_id);

    for (size_t b = 0; b < batch_size; b++) {
        size_t slot_id = slot_ids[b];
        auto seq_view = cache->slice(0, slot_id, slot_id + 1)
                            ->slice(1, 0, kv_seq_len)
                            ->contiguous()
                            ->view({kv_seq_len, cache->shape()[2], cache->shape()[3]});
        auto dst_view = flat->slice(0, b * kv_seq_len, (b + 1) * kv_seq_len);
        copy_contiguous_tensor_(dst_view, seq_view);
    }
    return flat;
}

// Scatter K/V from [slot_ids.size()*seq_len, nkvh, dh] into selected cache slots.
static void scatter_kv_to_cache_slots(const tensor_t& cache, const tensor_t& kv_flat,
                                      const std::vector<size_t>& slot_ids, size_t start_pos, size_t seq_len) {
    size_t batch_size = slot_ids.size();
    size_t maxseq = cache->shape()[1];
    size_t nkvh = cache->shape()[2];
    size_t dh = cache->shape()[3];
    size_t row_elems = seq_len * nkvh * dh;
    size_t bytes = row_elems * cache->elementSize();
    core::context().setDevice(cache->deviceType(), cache->deviceId());
    for (size_t b = 0; b < batch_size; b++) {
        size_t slot_id = slot_ids[b];
        size_t src_offset = b * row_elems * cache->elementSize();
        size_t dst_offset = ((slot_id * maxseq + start_pos) * nkvh * dh) * cache->elementSize();
        core::context().runtime().api()->memcpy_sync(
            cache->data() + dst_offset, kv_flat->data() + src_offset, bytes, LLAISYS_MEMCPY_D2D);
    }
}

static void ensure_kv_cache(Qwen2TPRankData& rd, size_t nlayer, size_t batch_size,
                             size_t maxseq, size_t nkvh_per_rank, size_t dh,
                             llaisysDataType_t dtype, llaisysDeviceType_t device_type, int dev_id) {
    rd.kv_caches.resize(nlayer);
    for (size_t i = 0; i < nlayer; i++) {
        core::context().setDevice(device_type, dev_id);
        rd.kv_caches[i].k_cache = Tensor::create(
            {batch_size, maxseq, nkvh_per_rank, dh}, dtype, device_type, dev_id);
        rd.kv_caches[i].v_cache = Tensor::create(
            {batch_size, maxseq, nkvh_per_rank, dh}, dtype, device_type, dev_id);
        rd.kv_caches[i].current_seq_len = 0;
    }
}

static tensor_t concat_dim0(const std::vector<tensor_t>& parts) {
    CHECK_ARGUMENT(!parts.empty(), "concat_dim0: no parts");
    size_t rows = 0;
    size_t cols = parts[0]->shape().size() > 1 ? parts[0]->shape()[1] : 0;
    llaisysDataType_t dtype = parts[0]->dtype();
    llaisysDeviceType_t device_type = parts[0]->deviceType();
    int device_id = parts[0]->deviceId();
    for (const auto& part : parts) {
        CHECK_ARGUMENT(part && part->isContiguous(), "concat_dim0: parts must be contiguous");
        CHECK_ARGUMENT(part->dtype() == dtype && part->deviceType() == device_type && part->deviceId() == device_id,
                       "concat_dim0: device/dtype mismatch");
        if (part->ndim() == 1) {
            rows += part->shape()[0];
        } else {
            CHECK_ARGUMENT(part->ndim() == 2 && part->shape()[1] == cols, "concat_dim0: shape mismatch");
            rows += part->shape()[0];
        }
    }

    tensor_t out = parts[0]->ndim() == 1
        ? Tensor::create({rows}, dtype, device_type, device_id)
        : Tensor::create({rows, cols}, dtype, device_type, device_id);
    size_t offset = 0;
    for (const auto& part : parts) {
        size_t n = part->shape()[0];
        tensor_t dst = out->slice(0, offset, offset + n);
        copy_contiguous_tensor_(dst, part);
        offset += n;
    }
    return out;
}

static void rebuild_decode_fused_weights(Qwen2Weights& w, size_t layer_idx) {
    const char* fused_qkv_env = std::getenv("LLAISYS_FUSED_QKV");
    const bool build_qkv = fused_qkv_env && std::atoi(fused_qkv_env) != 0;
    const char* fused_gate_env = std::getenv("LLAISYS_FUSED_GATE_UP");
    const bool build_gate_up = fused_gate_env && std::atoi(fused_gate_env) != 0;

    if (build_qkv && !w.attn_qkv_w[layer_idx] &&
        w.attn_q_w[layer_idx] && w.attn_k_w[layer_idx] && w.attn_v_w[layer_idx]) {
        w.attn_qkv_w[layer_idx] = concat_dim0({w.attn_q_w[layer_idx], w.attn_k_w[layer_idx], w.attn_v_w[layer_idx]});
    }
    if (build_qkv && !w.attn_qkv_b[layer_idx] &&
        w.attn_q_b[layer_idx] && w.attn_k_b[layer_idx] && w.attn_v_b[layer_idx]) {
        w.attn_qkv_b[layer_idx] = concat_dim0({w.attn_q_b[layer_idx], w.attn_k_b[layer_idx], w.attn_v_b[layer_idx]});
    }
    if (build_gate_up && !w.mlp_gate_up_w[layer_idx] && w.mlp_gate_w[layer_idx] && w.mlp_up_w[layer_idx]) {
        w.mlp_gate_up_w[layer_idx] = concat_dim0({w.mlp_gate_w[layer_idx], w.mlp_up_w[layer_idx]});
    }
}

Qwen2TPModel::Qwen2TPModel(const Qwen2Config& config, llaisysDeviceType_t device_type,
                            const std::vector<int>& device_ids)
    : config_(config), device_type_(device_type), device_ids_(device_ids),
      tp_size_(device_ids.size()), current_pos_(0), batch_size_(1),
      max_slots_(1), paged_kv_mode_(false), paged_block_size_(0),
      paged_max_blocks_(0), paged_max_blocks_per_slot_(0),
      prefill_scratch_slots_(0), paged_peak_used_blocks_(0),
      continuous_ready_(false)
#ifdef ENABLE_NVIDIA_API
      , decode_graph_ready_(false), decode_graph_batch_(0),
        decode_graph_(nullptr), decode_graph_exec_(nullptr)
#endif
      {

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
        ensure_kv_cache(rd, config_.nlayer, batch_size_, config_.maxseq,
                        nkvh_per_rank_, config_.dh, config_.dtype, device_type_, dev_id);

        auto& w = rd.weights;
        w.attn_norm_w.resize(config_.nlayer);
        w.attn_q_w.resize(config_.nlayer);
        w.attn_q_b.resize(config_.nlayer);
        w.attn_k_w.resize(config_.nlayer);
        w.attn_k_b.resize(config_.nlayer);
        w.attn_v_w.resize(config_.nlayer);
        w.attn_v_b.resize(config_.nlayer);
        w.attn_qkv_w.resize(config_.nlayer);
        w.attn_qkv_b.resize(config_.nlayer);
        w.attn_o_w.resize(config_.nlayer);
        w.mlp_norm_w.resize(config_.nlayer);
        w.mlp_gate_w.resize(config_.nlayer);
        w.mlp_up_w.resize(config_.nlayer);
        w.mlp_gate_up_w.resize(config_.nlayer);
        w.mlp_down_w.resize(config_.nlayer);
    }

#ifdef ENABLE_NVIDIA_API
    if (device_type_ == LLAISYS_DEVICE_NVIDIA && tp_size_ > 1) {
        nccl_comm_ = std::make_unique<device::nvidia::NcclComm>(device_ids_);
    }
#endif
}

Qwen2TPModel::~Qwen2TPModel() {
#ifdef ENABLE_NVIDIA_API
    if (decode_graph_exec_) cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(decode_graph_exec_));
    if (decode_graph_) cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(decode_graph_));
#endif
}

void Qwen2TPModel::reset_cache() {
    current_pos_ = 0;
    continuous_ready_ = false;
    for (auto& rd : ranks_) {
        for (auto& cache : rd.kv_caches) {
            cache.current_seq_len = 0;
        }
    }
    std::fill(slot_active_.begin(), slot_active_.end(), 0);
    std::fill(slot_seq_lens_.begin(), slot_seq_lens_.end(), 0);
}

void Qwen2TPModel::init_continuous(size_t max_slots) {
    CHECK_ARGUMENT(max_slots >= 1, "init_continuous: max_slots must be >= 1");
    paged_kv_mode_ = false;
    paged_block_size_ = 0;
    paged_max_blocks_ = 0;
    paged_max_blocks_per_slot_ = 0;
    prefill_scratch_slots_ = 0;
    slot_block_tables_.clear();
    free_paged_blocks_.clear();
    init_continuous_common(max_slots, max_slots);
}

void Qwen2TPModel::init_paged_continuous(size_t max_slots, size_t block_size,
                                         size_t max_blocks, size_t prefill_scratch_slots) {
    CHECK_ARGUMENT(tp_size_ == 1, "init_paged_continuous: TP=1 only in MVP");
    CHECK_ARGUMENT(device_type_ == LLAISYS_DEVICE_NVIDIA, "init_paged_continuous: CUDA only in MVP");
    CHECK_ARGUMENT(max_slots >= 1, "init_paged_continuous: max_slots must be >= 1");
    CHECK_ARGUMENT(block_size >= 1, "init_paged_continuous: block_size must be >= 1");
    CHECK_ARGUMENT(max_blocks >= 1, "init_paged_continuous: max_blocks must be >= 1");
    paged_kv_mode_ = true;
    paged_block_size_ = block_size;
    paged_max_blocks_ = max_blocks;
    paged_max_blocks_per_slot_ = (config_.maxseq + block_size - 1) / block_size;
    prefill_scratch_slots_ = std::max<size_t>(1, prefill_scratch_slots);
    prefill_scratch_slots_ = std::min(prefill_scratch_slots_, max_slots);
    slot_block_tables_.assign(max_slots, std::vector<int64_t>(paged_max_blocks_per_slot_, -1));
    reset_paged_blocks();

    init_continuous_common(max_slots, prefill_scratch_slots_);

    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        ensure_kv_cache(ranks_[rank], config_.nlayer, paged_max_blocks_, paged_block_size_,
                        nkvh_per_rank_, config_.dh, config_.dtype, device_type_, dev_id);
        ranks_[rank].paged_kv_caches.swap(ranks_[rank].kv_caches);
        ensure_kv_cache(ranks_[rank], config_.nlayer, prefill_scratch_slots_, config_.maxseq,
                        nkvh_per_rank_, config_.dh, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].block_tables = Tensor::create(
            {max_slots_, paged_max_blocks_per_slot_}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        sync_paged_block_table(rank);
    }
}

void Qwen2TPModel::init_continuous_common(size_t max_slots, size_t kv_slots) {
    max_slots_ = max_slots;
    batch_size_ = max_slots_;
    current_pos_ = 0;
    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        ensure_kv_cache(ranks_[rank], config_.nlayer, kv_slots, config_.maxseq,
                        nkvh_per_rank_, config_.dh, config_.dtype, device_type_, dev_id);
    }
    decode_meta_.resize(tp_size_);
#ifdef ENABLE_NVIDIA_API
    if (decode_graph_exec_) {
        cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(decode_graph_exec_));
        decode_graph_exec_ = nullptr;
    }
    if (decode_graph_) {
        cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(decode_graph_));
        decode_graph_ = nullptr;
    }
    decode_graph_ready_ = false;
    decode_graph_batch_ = 0;
#endif
    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        core::context().setDevice(device_type_, dev_id);
        decode_meta_[rank].slot_ids = Tensor::create({max_slots_}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        decode_meta_[rank].positions = Tensor::create({max_slots_}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        decode_meta_[rank].seq_lens = Tensor::create({max_slots_}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        decode_meta_[rank].pos_ids = Tensor::create({max_slots_}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        decode_meta_[rank].max_idx = Tensor::create({max_slots_}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        decode_meta_[rank].max_val = Tensor::create({max_slots_}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].partial_max_idx = Tensor::create(
            {max_slots_, kDecodeArgmaxParts}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        decode_meta_[rank].partial_max_val = Tensor::create(
            {max_slots_, kDecodeArgmaxParts}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].input_ids = Tensor::create({max_slots_}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        decode_meta_[rank].hidden_a = Tensor::create({max_slots_, config_.hs}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].hidden_b = Tensor::create({max_slots_, config_.hs}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].attn_norm = Tensor::create({max_slots_, config_.hs}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].qkv_flat = Tensor::create(
            {max_slots_, (nh_per_rank_ + 2 * nkvh_per_rank_) * config_.dh}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].q_flat = Tensor::create({max_slots_, nh_per_rank_ * config_.dh}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].k_flat = Tensor::create({max_slots_, nkvh_per_rank_ * config_.dh}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].v_flat = Tensor::create({max_slots_, nkvh_per_rank_ * config_.dh}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].attn_out = Tensor::create({max_slots_, nh_per_rank_, config_.dh}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].o_proj = Tensor::create({max_slots_, config_.hs}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].hidden_1 = Tensor::create({max_slots_, config_.hs}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].mlp_norm = Tensor::create({max_slots_, config_.hs}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].gate_up_out = Tensor::create({max_slots_, 2 * di_per_rank_}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].gate_out = Tensor::create({max_slots_, di_per_rank_}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].up_out = Tensor::create({max_slots_, di_per_rank_}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].swiglu_out = Tensor::create({max_slots_, di_per_rank_}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].mlp_out = Tensor::create({max_slots_, config_.hs}, config_.dtype, device_type_, dev_id);
        decode_meta_[rank].out_norm = Tensor::create({max_slots_, config_.hs}, config_.dtype, device_type_, dev_id);
        if (rank == 0) {
            decode_meta_[rank].logits = Tensor::create({max_slots_, config_.voc}, config_.dtype, device_type_, dev_id);
        }
    }
    slot_active_.assign(max_slots_, 0);
    slot_seq_lens_.assign(max_slots_, 0);
    slot_last_tokens_.assign(max_slots_, 0);
    continuous_ready_ = true;
}

int64_t Qwen2TPModel::prefill_slot(size_t slot_id, const std::vector<int64_t>& token_ids) {
    CHECK_ARGUMENT(continuous_ready_, "prefill_slot: init_continuous must be called first");
    CHECK_ARGUMENT(slot_id < max_slots_, "prefill_slot: invalid slot_id");
    CHECK_ARGUMENT(!token_ids.empty(), "prefill_slot: token_ids must not be empty");
    CHECK_ARGUMENT(token_ids.size() <= config_.maxseq, "prefill_slot: sequence length exceeds maxseq");

    std::vector<size_t> forward_slot_ids{paged_kv_mode_ ? 0 : slot_id};
    tensor_t logits = forward_slots(token_ids, forward_slot_ids, 0, token_ids.size());

    int dev_id = device_ids_[0];
    core::context().setDevice(device_type_, dev_id);
    tensor_t last_logits = logits->slice(0, token_ids.size() - 1, token_ids.size());
    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, dev_id);
    tensor_t max_val = Tensor::create({1}, config_.dtype, device_type_, dev_id);
    ops::argmax(max_idx, max_val, last_logits->view({config_.voc}));

    int64_t next_token = -1;
    std::vector<std::byte> buffer(sizeof(int64_t));
    core::context().runtime().api()->memcpy_sync(
        buffer.data(), max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
    std::memcpy(&next_token, buffer.data(), sizeof(int64_t));

    slot_active_[slot_id] = 1;
    slot_seq_lens_[slot_id] = token_ids.size();
    slot_last_tokens_[slot_id] = next_token;
    if (paged_kv_mode_) {
        release_paged_blocks(slot_id);
        allocate_paged_blocks_for_len(slot_id, token_ids.size());
        copy_prefill_to_paged_cache({slot_id}, {0}, token_ids.size());
    }
    return next_token;
}

int64_t Qwen2TPModel::prefill_slot_chunk(size_t slot_id, const std::vector<int64_t>& token_ids,
                                         bool final_chunk) {
    CHECK_ARGUMENT(continuous_ready_, "prefill_slot_chunk: init_continuous must be called first");
    CHECK_ARGUMENT(slot_id < max_slots_, "prefill_slot_chunk: invalid slot_id");
    CHECK_ARGUMENT(!token_ids.empty(), "prefill_slot_chunk: token_ids must not be empty");
    CHECK_ARGUMENT(slot_seq_lens_[slot_id] + token_ids.size() <= config_.maxseq,
                   "prefill_slot_chunk: sequence length exceeds maxseq");

    size_t start_pos = slot_seq_lens_[slot_id];
    if (paged_kv_mode_ && start_pos == 0) {
        release_paged_blocks(slot_id);
        slot_active_[slot_id] = 0;
    }
    size_t scratch_slot_id = paged_kv_mode_ ? 0 : slot_id;
    std::vector<size_t> slot_ids{scratch_slot_id};
    tensor_t logits = forward_slots(token_ids, slot_ids, start_pos, token_ids.size());

    slot_seq_lens_[slot_id] += token_ids.size();

    if (!final_chunk) {
        return -2;
    }

    int dev_id = device_ids_[0];
    core::context().setDevice(device_type_, dev_id);
    tensor_t last_logits = logits->slice(0, token_ids.size() - 1, token_ids.size());
    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, dev_id);
    tensor_t max_val = Tensor::create({1}, config_.dtype, device_type_, dev_id);
    ops::argmax(max_idx, max_val, last_logits->view({config_.voc}));

    int64_t next_token = -1;
    std::vector<std::byte> buffer(sizeof(int64_t));
    core::context().runtime().api()->memcpy_sync(
        buffer.data(), max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
    std::memcpy(&next_token, buffer.data(), sizeof(int64_t));
    slot_active_[slot_id] = 1;
    slot_last_tokens_[slot_id] = next_token;
    if (paged_kv_mode_) {
        allocate_paged_blocks_for_len(slot_id, slot_seq_lens_[slot_id]);
        copy_prefill_to_paged_cache({slot_id}, {scratch_slot_id}, slot_seq_lens_[slot_id]);
    }
    return next_token;
}

std::vector<int64_t> Qwen2TPModel::prefill_slots(const std::vector<size_t>& slot_ids,
                                                 const std::vector<int64_t>& token_ids,
                                                 size_t prompt_len) {
    CHECK_ARGUMENT(continuous_ready_, "prefill_slots: init_continuous must be called first");
    CHECK_ARGUMENT(!slot_ids.empty(), "prefill_slots: slot_ids must not be empty");
    CHECK_ARGUMENT(prompt_len > 0, "prefill_slots: prompt_len must be > 0");
    CHECK_ARGUMENT(prompt_len <= config_.maxseq, "prefill_slots: sequence length exceeds maxseq");
    CHECK_ARGUMENT(token_ids.size() == slot_ids.size() * prompt_len,
                   "prefill_slots: token count mismatch");

    for (size_t slot_id : slot_ids) {
        CHECK_ARGUMENT(slot_id < max_slots_, "prefill_slots: invalid slot_id");
    }

    CHECK_ARGUMENT(!paged_kv_mode_ || slot_ids.size() <= prefill_scratch_slots_,
                   "prefill_slots: batch exceeds paged scratch slots");
    std::vector<size_t> forward_slot_ids = slot_ids;
    if (paged_kv_mode_) {
        forward_slot_ids.resize(slot_ids.size());
        for (size_t i = 0; i < slot_ids.size(); i++) forward_slot_ids[i] = i;
    }
    tensor_t logits = forward_slots(token_ids, forward_slot_ids, 0, prompt_len);

    int dev_id = device_ids_[0];
    core::context().setDevice(device_type_, dev_id);
    std::vector<int64_t> next_tokens(slot_ids.size());
    for (size_t b = 0; b < slot_ids.size(); b++) {
        size_t last_idx = (b + 1) * prompt_len - 1;
        tensor_t last_logits = logits->slice(0, last_idx, last_idx + 1);
        tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        tensor_t max_val = Tensor::create({1}, config_.dtype, device_type_, dev_id);
        ops::argmax(max_idx, max_val, last_logits->view({config_.voc}));

        std::vector<std::byte> buffer(sizeof(int64_t));
        core::context().runtime().api()->memcpy_sync(
            buffer.data(), max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
        std::memcpy(&next_tokens[b], buffer.data(), sizeof(int64_t));

        size_t slot_id = slot_ids[b];
        slot_active_[slot_id] = 1;
        slot_seq_lens_[slot_id] = prompt_len;
        slot_last_tokens_[slot_id] = next_tokens[b];
    }
    if (paged_kv_mode_) {
        for (size_t slot_id : slot_ids) {
            release_paged_blocks(slot_id);
            allocate_paged_blocks_for_len(slot_id, prompt_len);
        }
        copy_prefill_to_paged_cache(slot_ids, forward_slot_ids, prompt_len);
    }
    return next_tokens;
}

std::vector<int64_t> Qwen2TPModel::prefill_slots_varlen(const std::vector<size_t>& slot_ids,
                                                        const std::vector<int64_t>& token_ids,
                                                        const std::vector<size_t>& prompt_lens,
                                                        size_t max_prompt_len) {
    CHECK_ARGUMENT(continuous_ready_, "prefill_slots_varlen: init_continuous must be called first");
    CHECK_ARGUMENT(paged_kv_mode_, "prefill_slots_varlen: paged KV only");
    CHECK_ARGUMENT(!slot_ids.empty(), "prefill_slots_varlen: slot_ids must not be empty");
    CHECK_ARGUMENT(slot_ids.size() == prompt_lens.size(), "prefill_slots_varlen: prompt_lens mismatch");
    CHECK_ARGUMENT(max_prompt_len > 0 && max_prompt_len <= config_.maxseq,
                   "prefill_slots_varlen: invalid max_prompt_len");
    CHECK_ARGUMENT(token_ids.size() == slot_ids.size() * max_prompt_len,
                   "prefill_slots_varlen: token count mismatch");
    CHECK_ARGUMENT(slot_ids.size() <= prefill_scratch_slots_,
                   "prefill_slots_varlen: batch exceeds paged scratch slots");

    std::vector<size_t> scratch_slot_ids(slot_ids.size());
    for (size_t i = 0; i < slot_ids.size(); i++) {
        CHECK_ARGUMENT(slot_ids[i] < max_slots_, "prefill_slots_varlen: invalid slot_id");
        CHECK_ARGUMENT(prompt_lens[i] > 0 && prompt_lens[i] <= max_prompt_len,
                       "prefill_slots_varlen: invalid prompt length");
        scratch_slot_ids[i] = i;
    }

    tensor_t logits = forward_slots(token_ids, scratch_slot_ids, 0, max_prompt_len);

    int dev_id = device_ids_[0];
    core::context().setDevice(device_type_, dev_id);
    std::vector<int64_t> next_tokens(slot_ids.size());
    for (size_t b = 0; b < slot_ids.size(); b++) {
        size_t last_idx = b * max_prompt_len + prompt_lens[b] - 1;
        tensor_t last_logits = logits->slice(0, last_idx, last_idx + 1);
        tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        tensor_t max_val = Tensor::create({1}, config_.dtype, device_type_, dev_id);
        ops::argmax(max_idx, max_val, last_logits->view({config_.voc}));

        std::vector<std::byte> buffer(sizeof(int64_t));
        core::context().runtime().api()->memcpy_sync(
            buffer.data(), max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
        std::memcpy(&next_tokens[b], buffer.data(), sizeof(int64_t));

        size_t slot_id = slot_ids[b];
        release_paged_blocks(slot_id);
        allocate_paged_blocks_for_len(slot_id, prompt_lens[b]);
        copy_prefill_to_paged_cache({slot_id}, {scratch_slot_ids[b]}, prompt_lens[b]);
        slot_active_[slot_id] = 1;
        slot_seq_lens_[slot_id] = prompt_lens[b];
        slot_last_tokens_[slot_id] = next_tokens[b];
    }
    return next_tokens;
}

std::vector<int64_t> Qwen2TPModel::decode_slots(const std::vector<size_t>& slot_ids,
                                                const std::vector<int64_t>& input_tokens) {
    CHECK_ARGUMENT(continuous_ready_, "decode_slots: init_continuous must be called first");
    CHECK_ARGUMENT(!slot_ids.empty(), "decode_slots: slot_ids must not be empty");
    CHECK_ARGUMENT(slot_ids.size() == input_tokens.size(), "decode_slots: input size mismatch");

    for (size_t i = 0; i < slot_ids.size(); i++) {
        size_t slot_id = slot_ids[i];
        CHECK_ARGUMENT(slot_id < max_slots_, "decode_slots: invalid slot_id");
        CHECK_ARGUMENT(slot_active_[slot_id], "decode_slots: inactive slot");
        CHECK_ARGUMENT(slot_seq_lens_[slot_id] + 1 <= config_.maxseq,
                       "decode_slots: sequence length exceeds maxseq");
    }

    std::vector<size_t> start_positions(slot_ids.size());
    for (size_t i = 0; i < slot_ids.size(); i++) start_positions[i] = slot_seq_lens_[slot_ids[i]];
    if (paged_kv_mode_) {
        for (size_t i = 0; i < slot_ids.size(); i++) {
            ensure_paged_position_allocated(slot_ids[i], start_positions[i]);
        }
        for (int rank = 0; rank < tp_size_; rank++) sync_paged_block_table(rank);
    }

    tensor_t logits = forward_slots_decode(input_tokens, slot_ids, start_positions);

    int dev_id = device_ids_[0];
    core::context().setDevice(device_type_, dev_id);
    tensor_t max_idx = decode_meta_[0].max_idx->slice(0, 0, slot_ids.size());
    tensor_t max_val = decode_meta_[0].max_val->slice(0, 0, slot_ids.size());
    const char* fast_argmax_env = std::getenv("LLAISYS_FAST_ARGMAX");
    const bool use_fast_argmax = fast_argmax_env && std::atoi(fast_argmax_env) != 0;
    if (use_fast_argmax && device_type_ == LLAISYS_DEVICE_NVIDIA) {
        tensor_t partial_idx = decode_meta_[0].partial_max_idx->slice(0, 0, slot_ids.size());
        tensor_t partial_val = decode_meta_[0].partial_max_val->slice(0, 0, slot_ids.size());
        ops::argmax_batch_fast(max_idx, max_val, logits, partial_idx, partial_val);
    } else {
        ops::argmax_batch(max_idx, max_val, logits);
    }

    std::vector<int64_t> next_tokens(slot_ids.size());
    core::context().runtime().api()->memcpy_sync(
        next_tokens.data(), max_idx->data(), sizeof(int64_t) * slot_ids.size(), LLAISYS_MEMCPY_D2H);
    for (size_t b = 0; b < slot_ids.size(); b++) {
        size_t slot_id = slot_ids[b];
        slot_seq_lens_[slot_id] += 1;
        slot_last_tokens_[slot_id] = next_tokens[b];
    }
    return next_tokens;
}

void Qwen2TPModel::release_slot(size_t slot_id) {
    CHECK_ARGUMENT(continuous_ready_, "release_slot: init_continuous must be called first");
    CHECK_ARGUMENT(slot_id < max_slots_, "release_slot: invalid slot_id");
    slot_active_[slot_id] = 0;
    slot_seq_lens_[slot_id] = 0;
    slot_last_tokens_[slot_id] = 0;
    release_paged_blocks(slot_id);
}

size_t Qwen2TPModel::slot_seq_len(size_t slot_id) const {
    CHECK_ARGUMENT(slot_id < slot_seq_lens_.size(), "slot_seq_len: invalid slot_id");
    return slot_seq_lens_[slot_id];
}

void Qwen2TPModel::reset_paged_blocks() {
    free_paged_blocks_.clear();
    free_paged_blocks_.reserve(paged_max_blocks_);
    for (size_t i = 0; i < paged_max_blocks_; i++) {
        free_paged_blocks_.push_back(paged_max_blocks_ - 1 - i);
    }
    paged_peak_used_blocks_ = 0;
    for (auto& table : slot_block_tables_) {
        std::fill(table.begin(), table.end(), -1);
    }
}

void Qwen2TPModel::release_paged_blocks(size_t slot_id) {
    if (!paged_kv_mode_ || slot_id >= slot_block_tables_.size()) return;
    for (auto& block : slot_block_tables_[slot_id]) {
        if (block >= 0) {
            free_paged_blocks_.push_back(static_cast<size_t>(block));
            block = -1;
        }
    }
    for (int rank = 0; rank < tp_size_; rank++) sync_paged_block_table(rank);
}

void Qwen2TPModel::ensure_paged_position_allocated(size_t slot_id, size_t position) {
    CHECK_ARGUMENT(paged_kv_mode_, "paged KV is not enabled");
    CHECK_ARGUMENT(slot_id < slot_block_tables_.size(), "invalid paged slot");
    size_t logical_block = position / paged_block_size_;
    CHECK_ARGUMENT(logical_block < paged_max_blocks_per_slot_, "paged position exceeds maxseq");
    if (slot_block_tables_[slot_id][logical_block] >= 0) return;
    CHECK_ARGUMENT(!free_paged_blocks_.empty(), "paged KV block OOM");
    size_t physical_block = free_paged_blocks_.back();
    free_paged_blocks_.pop_back();
    slot_block_tables_[slot_id][logical_block] = static_cast<int64_t>(physical_block);
    size_t used = paged_max_blocks_ - free_paged_blocks_.size();
    paged_peak_used_blocks_ = std::max(paged_peak_used_blocks_, used);
}

void Qwen2TPModel::allocate_paged_blocks_for_len(size_t slot_id, size_t seq_len) {
    for (size_t pos = 0; pos < seq_len; pos += paged_block_size_) {
        ensure_paged_position_allocated(slot_id, pos);
    }
}

void Qwen2TPModel::sync_paged_block_table(int rank) {
    if (!paged_kv_mode_) return;
    int dev_id = device_ids_[rank];
    core::context().setDevice(device_type_, dev_id);
    std::vector<int64_t> flat(max_slots_ * paged_max_blocks_per_slot_, -1);
    for (size_t s = 0; s < max_slots_; s++) {
        std::copy(slot_block_tables_[s].begin(), slot_block_tables_[s].end(),
                  flat.begin() + s * paged_max_blocks_per_slot_);
    }
    decode_meta_[rank].block_tables->load(flat.data());
}

PagedKVStats Qwen2TPModel::paged_kv_stats() const {
    PagedKVStats stats{};
    stats.block_size = paged_block_size_;
    stats.max_blocks = paged_max_blocks_;
    stats.free_blocks = free_paged_blocks_.size();
    stats.used_blocks = paged_max_blocks_ >= stats.free_blocks ? paged_max_blocks_ - stats.free_blocks : 0;
    stats.peak_used_blocks = paged_peak_used_blocks_;
    stats.kv_capacity_tokens = paged_max_blocks_ * paged_block_size_;
    return stats;
}

void Qwen2TPModel::copy_prefill_to_paged_cache(const std::vector<size_t>& real_slot_ids,
                                               const std::vector<size_t>& scratch_slot_ids,
                                               size_t prompt_len) {
    if (!paged_kv_mode_) return;
    CHECK_ARGUMENT(real_slot_ids.size() == scratch_slot_ids.size(), "paged copy slot mismatch");
    std::vector<int64_t> real_i64(real_slot_ids.size());
    std::vector<int64_t> scratch_i64(scratch_slot_ids.size());
    for (size_t i = 0; i < real_slot_ids.size(); i++) {
        real_i64[i] = static_cast<int64_t>(real_slot_ids[i]);
        scratch_i64[i] = static_cast<int64_t>(scratch_slot_ids[i]);
    }
    for (int rank = 0; rank < tp_size_; rank++) {
        sync_paged_block_table(rank);
        int dev_id = device_ids_[rank];
        core::context().setDevice(device_type_, dev_id);
        tensor_t real_t = Tensor::create({real_i64.size()}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        tensor_t scratch_t = Tensor::create({scratch_i64.size()}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        real_t->load(real_i64.data());
        scratch_t->load(scratch_i64.data());
        for (size_t layer = 0; layer < config_.nlayer; layer++) {
            ops::copy_kv_slots_to_blocks(
                ranks_[rank].paged_kv_caches[layer].k_cache,
                ranks_[rank].paged_kv_caches[layer].v_cache,
                ranks_[rank].kv_caches[layer].k_cache,
                ranks_[rank].kv_caches[layer].v_cache,
                decode_meta_[rank].block_tables, real_t, scratch_t, prompt_len);
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

    rebuild_decode_fused_weights(w, layer_idx);
}

void Qwen2TPModel::allreduce_sum(std::vector<tensor_t>& tensors) {
    if (tp_size_ <= 1) return;

#ifdef ENABLE_NVIDIA_API
    if (device_type_ == LLAISYS_DEVICE_NVIDIA && nccl_comm_) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<void*> bufs(tp_size_);
        for (int i = 0; i < tp_size_; i++) {
            bufs[i] = tensors[i]->data();
        }
        nccl_comm_->allreduceSum(bufs, tensors[0]->numel(), tensors[0]->dtype());
        if (env_flag_enabled("LLAISYS_TP_TIMING")) {
            std::cerr << "[TP timing] allreduce numel=" << tensors[0]->numel()
                      << " ms=" << ms_since(t0) << std::endl;
        }
    }
#endif
}

int64_t Qwen2TPModel::infer(const std::vector<int64_t>& token_ids) {
    auto result = infer_batch(token_ids, 1);
    return result[0];
}

std::vector<int64_t> Qwen2TPModel::infer_batch(const std::vector<int64_t>& token_ids, size_t batch_size) {
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

    if (batch_size != batch_size_) {
        batch_size_ = batch_size;
        for (int rank = 0; rank < tp_size_; rank++) {
            int dev_id = device_ids_[rank];
            ensure_kv_cache(ranks_[rank], config_.nlayer, batch_size_, config_.maxseq,
                            nkvh_per_rank_, config_.dh, config_.dtype, device_type_, dev_id);
        }
        current_pos_ = 0;
        start_pos = 0;
    }

    std::vector<int64_t> new_tokens_vec;
    new_tokens_vec.reserve(batch_size * seq_len);
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < seq_len; i++) {
            new_tokens_vec.push_back(token_ids[b * total_len + start_pos + i]);
        }
    }

    tensor_t logits = forward(new_tokens_vec, start_pos, seq_len, batch_size);

    current_pos_ = total_len;

    int dev_id = device_ids_[0];
    core::context().setDevice(device_type_, dev_id);

    std::vector<int64_t> next_tokens(batch_size);
    for (size_t b = 0; b < batch_size; b++) {
        size_t last_idx = (b + 1) * seq_len - 1;
        tensor_t last_logits = logits->slice(0, last_idx, last_idx + 1);

        tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        tensor_t max_val = Tensor::create({1}, config_.dtype, device_type_, dev_id);
        ops::argmax(max_idx, max_val, last_logits->view({config_.voc}));

        std::vector<std::byte> buffer(sizeof(int64_t));
        core::context().runtime().api()->memcpy_sync(
            buffer.data(), max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
        std::memcpy(&next_tokens[b], buffer.data(), sizeof(int64_t));
    }

    return next_tokens;
}

tensor_t Qwen2TPModel::forward(const std::vector<int64_t>& new_tokens, size_t start_pos, size_t seq_len, size_t batch_size) {
    std::vector<size_t> slot_ids(batch_size);
    for (size_t i = 0; i < batch_size; i++) slot_ids[i] = i;
    return forward_slots(new_tokens, slot_ids, start_pos, seq_len);
}

tensor_t Qwen2TPModel::forward_slots(const std::vector<int64_t>& new_tokens, const std::vector<size_t>& slot_ids,
                                     size_t start_pos, size_t seq_len) {
    size_t batch_size = slot_ids.size();
    CHECK_ARGUMENT(new_tokens.size() == batch_size * seq_len, "forward_slots: token count mismatch");
    const bool timing = env_flag_enabled("LLAISYS_PREFILL_TIMING");
    auto total_t0 = std::chrono::steady_clock::now();
    auto phase_t0 = total_t0;
    std::vector<tensor_t> hidden_states(tp_size_);

    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        core::context().setDevice(device_type_, dev_id);

        tensor_t input_ids = Tensor::create({batch_size * seq_len}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        input_ids->load(new_tokens.data());

        hidden_states[rank] = Tensor::create({batch_size * seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
        ops::embedding(hidden_states[rank], input_ids, ranks_[rank].weights.in_embed);
    }
    if (timing) {
        sync_device_for_timing(device_type_);
        std::cerr << "[prefill timing] batch=" << batch_size << " seq_len=" << seq_len
                  << " start_pos=" << start_pos
                  << " embedding_ms=" << ms_since(phase_t0);
        phase_t0 = std::chrono::steady_clock::now();
    }

    for (size_t layer = 0; layer < config_.nlayer; layer++) {
        apply_layer_slots(layer, hidden_states, slot_ids, start_pos, seq_len);
    }
    if (timing) {
        sync_device_for_timing(device_type_);
        std::cerr << " layers_ms=" << ms_since(phase_t0);
        phase_t0 = std::chrono::steady_clock::now();
    }

    int dev_id = device_ids_[0];
    core::context().setDevice(device_type_, dev_id);

    tensor_t normed = Tensor::create({batch_size * seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
    ops::rms_norm(normed, hidden_states[0], ranks_[0].weights.out_norm_w, config_.epsilon);

    tensor_t logits = Tensor::create({batch_size * seq_len, config_.voc}, config_.dtype, device_type_, dev_id);
    ops::linear(logits, normed, ranks_[0].weights.out_embed, nullptr);
    if (timing) {
        sync_device_for_timing(device_type_);
        std::cerr << " lm_head_ms=" << ms_since(phase_t0)
                  << " total_ms=" << ms_since(total_t0) << std::endl;
    }

    return logits;
}

tensor_t Qwen2TPModel::forward_slots_decode(const std::vector<int64_t>& new_tokens,
                                            const std::vector<size_t>& slot_ids,
                                            const std::vector<size_t>& start_positions) {
    size_t batch_size = slot_ids.size();
    CHECK_ARGUMENT(new_tokens.size() == batch_size, "forward_slots_decode: token count mismatch");
    CHECK_ARGUMENT(start_positions.size() == batch_size, "forward_slots_decode: position count mismatch");
    std::vector<int64_t> pos_ids_vec(batch_size);
    std::vector<int64_t> slot_ids_i64(batch_size);
    std::vector<int64_t> seq_lens_i64(batch_size);
    for (size_t b = 0; b < batch_size; b++) {
        pos_ids_vec[b] = static_cast<int64_t>(start_positions[b]);
        slot_ids_i64[b] = static_cast<int64_t>(slot_ids[b]);
        seq_lens_i64[b] = static_cast<int64_t>(start_positions[b]);
    }

    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        core::context().setDevice(device_type_, dev_id);
        auto& meta = decode_meta_[rank];
        tensor_t slot_ids_t = meta.slot_ids->slice(0, 0, batch_size);
        tensor_t positions_t = meta.positions->slice(0, 0, batch_size);
        tensor_t seq_lens_t = meta.seq_lens->slice(0, 0, batch_size);
        tensor_t pos_ids_t = meta.pos_ids->slice(0, 0, batch_size);
        slot_ids_t->load(slot_ids_i64.data());
        positions_t->load(pos_ids_vec.data());
        seq_lens_t->load(seq_lens_i64.data());
        pos_ids_t->load(pos_ids_vec.data());

        tensor_t input_ids = meta.input_ids->slice(0, 0, batch_size);
        input_ids->load(new_tokens.data());
    }

#ifdef ENABLE_NVIDIA_API
    const char* graph_env = std::getenv("LLAISYS_ENABLE_DECODE_GRAPH");
    const bool graph_enabled = graph_env && std::atoi(graph_env) != 0;
    const bool can_use_graph = graph_enabled &&
                               device_type_ == LLAISYS_DEVICE_NVIDIA &&
                               tp_size_ == 1 &&
                               is_decode_graph_bucket(batch_size) &&
                               !env_flag_enabled("LLAISYS_DECODE_TIMING");
    if (can_use_graph && decode_graph_ready_ && decode_graph_batch_ != batch_size) {
        cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(decode_graph_exec_));
        cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(decode_graph_));
        decode_graph_exec_ = nullptr;
        decode_graph_ = nullptr;
        decode_graph_ready_ = false;
        decode_graph_batch_ = 0;
    }
    if (can_use_graph && decode_graph_ready_ && decode_graph_batch_ == batch_size) {
        cudaError_t err = cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(decode_graph_exec_), 0);
        if (err == cudaSuccess) {
            return decode_meta_[0].logits->slice(0, 0, batch_size);
        }
        std::cerr << "[TP] decode CUDA graph replay failed: "
                  << cudaGetErrorString(err) << "; falling back to normal decode" << std::endl;
        decode_graph_ready_ = false;
    }

    if (can_use_graph && !decode_graph_ready_) {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        tensor_t logits;
        bool graph_ok = true;
        cudaError_t err = cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            graph_ok = false;
        }
        try {
            if (graph_ok) logits = forward_slots_decode_compute(batch_size, slot_ids, start_positions);
        } catch (const std::exception& e) {
            std::cerr << "[TP] decode CUDA graph capture disabled: " << e.what() << std::endl;
            graph_ok = false;
        }
        if (graph_ok) {
            err = cudaStreamEndCapture(0, &graph);
            graph_ok = (err == cudaSuccess);
        } else {
            cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(0, &status) == cudaSuccess && status != cudaStreamCaptureStatusNone) {
                cudaStreamEndCapture(0, &graph);
                if (graph) cudaGraphDestroy(graph);
                graph = nullptr;
            }
        }
        if (graph_ok) {
            err = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
            graph_ok = (err == cudaSuccess);
        }
        if (graph_ok) {
            decode_graph_ = graph;
            decode_graph_exec_ = graph_exec;
            decode_graph_ready_ = true;
            decode_graph_batch_ = batch_size;
            return logits;
        }
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        decode_graph_ready_ = false;
        decode_graph_batch_ = 0;
        static bool graph_warned = false;
        if (!graph_warned) {
            std::cerr << "[TP] decode CUDA graph capture unavailable; falling back to normal decode" << std::endl;
            graph_warned = true;
        }
    }
#endif

    return forward_slots_decode_compute(batch_size, slot_ids, start_positions);
}

tensor_t Qwen2TPModel::forward_slots_decode_compute(size_t batch_size,
                                                    const std::vector<size_t>& slot_ids,
                                                    const std::vector<size_t>& start_positions) {
    const bool timing = env_flag_enabled("LLAISYS_DECODE_TIMING");
    auto total_t0 = std::chrono::steady_clock::now();
    auto phase_t0 = total_t0;
    std::vector<tensor_t> hidden_states(tp_size_);
    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        core::context().setDevice(device_type_, dev_id);
        auto& meta = decode_meta_[rank];
        tensor_t input_ids = meta.input_ids->slice(0, 0, batch_size);
        hidden_states[rank] = meta.hidden_a->slice(0, 0, batch_size);
        ops::embedding(hidden_states[rank], input_ids, ranks_[rank].weights.in_embed);
    }
    if (timing) {
        sync_device_for_timing(device_type_);
        std::cerr << "[decode timing] batch=" << batch_size
                  << " embedding_ms=" << ms_since(phase_t0);
        phase_t0 = std::chrono::steady_clock::now();
    }

    for (size_t layer = 0; layer < config_.nlayer; layer++) {
        apply_layer_slots_decode(layer, hidden_states, slot_ids, start_positions);
    }
    if (timing) {
        sync_device_for_timing(device_type_);
        std::cerr << " layers_ms=" << ms_since(phase_t0);
        phase_t0 = std::chrono::steady_clock::now();
    }

    int dev_id = device_ids_[0];
    core::context().setDevice(device_type_, dev_id);

    tensor_t normed = decode_meta_[0].out_norm->slice(0, 0, batch_size);
    ops::rms_norm(normed, hidden_states[0], ranks_[0].weights.out_norm_w, config_.epsilon);

    tensor_t logits = decode_meta_[0].logits->slice(0, 0, batch_size);
    ops::linear(logits, normed, ranks_[0].weights.out_embed, nullptr);
    if (timing) {
        sync_device_for_timing(device_type_);
        std::cerr << " lm_head_ms=" << ms_since(phase_t0)
                  << " total_ms=" << ms_since(total_t0) << std::endl;
    }

    return logits;
}

void Qwen2TPModel::apply_layer(size_t layer_idx, std::vector<tensor_t>& hidden_states,
                                size_t start_pos, size_t seq_len, size_t batch_size) {
    std::vector<size_t> slot_ids(batch_size);
    for (size_t i = 0; i < batch_size; i++) slot_ids[i] = i;
    apply_layer_slots(layer_idx, hidden_states, slot_ids, start_pos, seq_len);
}

void Qwen2TPModel::apply_layer_slots(size_t layer_idx, std::vector<tensor_t>& hidden_states,
                                     const std::vector<size_t>& slot_ids, size_t start_pos, size_t seq_len) {
    size_t batch_size = slot_ids.size();
    size_t kv_seq_len = start_pos + seq_len;
    size_t flat_seq_len = batch_size * seq_len;
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
            {flat_seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
        ops::rms_norm(attn_norm_out, hidden_states[rank], w.attn_norm_w[layer_idx], config_.epsilon);

        // Q/K/V projections (column-parallel: sharded output dim)
        tensor_t q_flat = Tensor::create(
            {flat_seq_len, nh_per_rank_ * config_.dh}, config_.dtype, device_type_, dev_id);
        tensor_t k_flat = Tensor::create(
            {flat_seq_len, nkvh_per_rank_ * config_.dh}, config_.dtype, device_type_, dev_id);
        tensor_t v_flat = Tensor::create(
            {flat_seq_len, nkvh_per_rank_ * config_.dh}, config_.dtype, device_type_, dev_id);

        const char* fused_qkv_env = std::getenv("LLAISYS_FUSED_QKV");
        const bool use_fused_qkv = fused_qkv_env && std::atoi(fused_qkv_env) != 0 &&
                                   w.attn_qkv_w[layer_idx] && w.attn_qkv_b[layer_idx];
        if (use_fused_qkv) {
            tensor_t qkv_flat = Tensor::create(
                {flat_seq_len, (nh_per_rank_ + 2 * nkvh_per_rank_) * config_.dh},
                config_.dtype, device_type_, dev_id);
            ops::linear(qkv_flat, attn_norm_out, w.attn_qkv_w[layer_idx], w.attn_qkv_b[layer_idx]);
            ops::split_qkv_decode(q_flat, k_flat, v_flat, qkv_flat);
        } else {
            ops::linear(q_flat, attn_norm_out, w.attn_q_w[layer_idx], w.attn_q_b[layer_idx]);
            ops::linear(k_flat, attn_norm_out, w.attn_k_w[layer_idx], w.attn_k_b[layer_idx]);
            ops::linear(v_flat, attn_norm_out, w.attn_v_w[layer_idx], w.attn_v_b[layer_idx]);
        }

        tensor_t q = q_flat->view({flat_seq_len, nh_per_rank_, config_.dh});
        tensor_t k = k_flat->view({flat_seq_len, nkvh_per_rank_, config_.dh});
        tensor_t v = v_flat->view({flat_seq_len, nkvh_per_rank_, config_.dh});

        // RoPE
        std::vector<int64_t> pos_ids_vec(flat_seq_len);
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t i = 0; i < seq_len; i++)
                pos_ids_vec[b * seq_len + i] = start_pos + i;
        }
        tensor_t pos_ids = Tensor::create({flat_seq_len}, LLAISYS_DTYPE_I64, device_type_, dev_id);
        pos_ids->load(pos_ids_vec.data());
        ops::rope(q, q, pos_ids, config_.theta);
        ops::rope(k, k, pos_ids, config_.theta);

        // KV cache update
        CHECK_ARGUMENT(kv_seq_len <= config_.maxseq, "kv_seq_len exceeds maxseq");
        auto& kv_cache = ranks_[rank].kv_caches[layer_idx];
        scatter_kv_to_cache_slots(kv_cache.k_cache, k, slot_ids, start_pos, seq_len);
        scatter_kv_to_cache_slots(kv_cache.v_cache, v, slot_ids, start_pos, seq_len);
        kv_cache.current_seq_len = kv_seq_len;

        // Gather full K/V from cache
        tensor_t full_k = gather_kv_from_cache_slots(kv_cache.k_cache, slot_ids, kv_seq_len,
                                                     config_.dtype, device_type_, dev_id);
        tensor_t full_v = gather_kv_from_cache_slots(kv_cache.v_cache, slot_ids, kv_seq_len,
                                                     config_.dtype, device_type_, dev_id);

        // Self-attention
        tensor_t attn_out = Tensor::create(
            {flat_seq_len, nh_per_rank_, config_.dh}, config_.dtype, device_type_, dev_id);
        ops::self_attention(attn_out, q, full_k, full_v, scale, seq_len);

        // O projection (row-parallel: sharded input dim)
        tensor_t attn_out_flat = attn_out->view({flat_seq_len, nh_per_rank_ * config_.dh});
        o_proj[rank] = Tensor::create({flat_seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
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
            {flat_seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
        ops::add(hidden_states_1[rank], hidden_states[rank], o_proj[rank]);

        // MLP norm
        tensor_t mlp_norm_out = Tensor::create(
            {flat_seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
        ops::rms_norm(mlp_norm_out, hidden_states_1[rank], w.mlp_norm_w[layer_idx], config_.epsilon);

        // Gate/Up projections (column-parallel)
        tensor_t gate_out = Tensor::create(
            {flat_seq_len, di_per_rank_}, config_.dtype, device_type_, dev_id);
        tensor_t up_out = Tensor::create(
            {flat_seq_len, di_per_rank_}, config_.dtype, device_type_, dev_id);
        const char* fused_gate_env = std::getenv("LLAISYS_FUSED_GATE_UP");
        const bool use_fused_gate_up = fused_gate_env && std::atoi(fused_gate_env) != 0 &&
                                       w.mlp_gate_up_w[layer_idx];
        if (use_fused_gate_up) {
            tensor_t gate_up_out = Tensor::create(
                {flat_seq_len, 2 * di_per_rank_}, config_.dtype, device_type_, dev_id);
            ops::linear(gate_up_out, mlp_norm_out, w.mlp_gate_up_w[layer_idx], nullptr);
            ops::split_gate_up_decode(gate_out, up_out, gate_up_out);
        } else {
            ops::linear(gate_out, mlp_norm_out, w.mlp_gate_w[layer_idx], nullptr);
            ops::linear(up_out, mlp_norm_out, w.mlp_up_w[layer_idx], nullptr);
        }

        // SwiGLU
        tensor_t swiglu_out = Tensor::create(
            {flat_seq_len, di_per_rank_}, config_.dtype, device_type_, dev_id);
        ops::swiglu(swiglu_out, gate_out, up_out);

        // Down projection (row-parallel)
        mlp_out[rank] = Tensor::create(
            {flat_seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
        ops::linear(mlp_out[rank], swiglu_out, w.mlp_down_w[layer_idx], nullptr);
    }

    // All-reduce after Down projection
    allreduce_sum(mlp_out);

    // Phase 3: Final residual
    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        core::context().setDevice(device_type_, dev_id);

        tensor_t output = Tensor::create(
            {flat_seq_len, config_.hs}, config_.dtype, device_type_, dev_id);
        ops::add(output, hidden_states_1[rank], mlp_out[rank]);
        hidden_states[rank] = output;
    }
}

void Qwen2TPModel::apply_layer_slots_decode(size_t layer_idx, std::vector<tensor_t>& hidden_states,
                                            const std::vector<size_t>& slot_ids,
                                            const std::vector<size_t>& start_positions) {
    size_t batch_size = slot_ids.size();
    float scale = 1.0f / std::sqrt(static_cast<float>(config_.dh));

    std::vector<tensor_t> o_proj(tp_size_);
    std::vector<tensor_t> hidden_states_1(tp_size_);
    std::vector<tensor_t> mlp_out(tp_size_);

    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        core::context().setDevice(device_type_, dev_id);
        auto& w = ranks_[rank].weights;
        auto& meta = decode_meta_[rank];

        tensor_t attn_norm_out = meta.attn_norm->slice(0, 0, batch_size);
        ops::rms_norm(attn_norm_out, hidden_states[rank], w.attn_norm_w[layer_idx], config_.epsilon);

        tensor_t q_flat = meta.q_flat->slice(0, 0, batch_size);
        tensor_t k_flat = meta.k_flat->slice(0, 0, batch_size);
        tensor_t v_flat = meta.v_flat->slice(0, 0, batch_size);

        const char* fused_qkv_env = std::getenv("LLAISYS_FUSED_QKV");
        const bool use_fused_qkv = fused_qkv_env && std::atoi(fused_qkv_env) != 0 &&
                                   w.attn_qkv_w[layer_idx] && w.attn_qkv_b[layer_idx];
        if (use_fused_qkv) {
            tensor_t qkv_flat = meta.qkv_flat->slice(0, 0, batch_size);
            ops::linear(qkv_flat, attn_norm_out, w.attn_qkv_w[layer_idx], w.attn_qkv_b[layer_idx]);
            ops::split_qkv_decode(q_flat, k_flat, v_flat, qkv_flat);
        } else {
            ops::linear(q_flat, attn_norm_out, w.attn_q_w[layer_idx], w.attn_q_b[layer_idx]);
            ops::linear(k_flat, attn_norm_out, w.attn_k_w[layer_idx], w.attn_k_b[layer_idx]);
            ops::linear(v_flat, attn_norm_out, w.attn_v_w[layer_idx], w.attn_v_b[layer_idx]);
        }

        tensor_t q = q_flat->view({batch_size, nh_per_rank_, config_.dh});
        tensor_t k = k_flat->view({batch_size, nkvh_per_rank_, config_.dh});
        tensor_t v = v_flat->view({batch_size, nkvh_per_rank_, config_.dh});

        for (size_t pos : start_positions) {
            CHECK_ARGUMENT(pos + 1 <= config_.maxseq, "decode kv_seq_len exceeds maxseq");
        }
        auto& kv_cache = paged_kv_mode_ ? ranks_[rank].paged_kv_caches[layer_idx]
                                        : ranks_[rank].kv_caches[layer_idx];
        tensor_t pos_ids = meta.pos_ids->slice(0, 0, batch_size);
        tensor_t slot_ids_t = meta.slot_ids->slice(0, 0, batch_size);
        tensor_t positions_t = meta.positions->slice(0, 0, batch_size);
        tensor_t seq_lens_t = meta.seq_lens->slice(0, 0, batch_size);
        if (paged_kv_mode_) {
            ops::rope_and_scatter_kv_paged_decode(q, k, v, kv_cache.k_cache, kv_cache.v_cache,
                                                  meta.block_tables, pos_ids, slot_ids_t,
                                                  positions_t, config_.theta);
        } else {
            const char* fused_env = std::getenv("LLAISYS_FUSED_ROPE_SCATTER");
            const bool use_fused = fused_env && std::atoi(fused_env) != 0;
            if (use_fused) {
                ops::rope_and_scatter_kv_decode(q, k, v, kv_cache.k_cache, kv_cache.v_cache,
                                                pos_ids, slot_ids_t, positions_t, config_.theta);
            } else {
                ops::rope(q, q, pos_ids, config_.theta);
                ops::rope(k, k, pos_ids, config_.theta);
                ops::scatter_kv_decode(kv_cache.k_cache, kv_cache.v_cache, k, v, slot_ids_t, positions_t);
            }
        }
        kv_cache.current_seq_len = 0;
        for (size_t pos : start_positions) {
            kv_cache.current_seq_len = std::max(kv_cache.current_seq_len, pos + 1);
        }

        tensor_t attn_out = meta.attn_out->slice(0, 0, batch_size);
        if (paged_kv_mode_) {
            ops::self_attention_paged_slots_decode(attn_out, q, kv_cache.k_cache, kv_cache.v_cache,
                                                   meta.block_tables, slot_ids_t, seq_lens_t, scale);
        } else if (env_flag_enabled("LLAISYS_GQA_ATTN")) {
            ops::self_attention_gqa_slots_decode(attn_out, q, kv_cache.k_cache, kv_cache.v_cache,
                                                 slot_ids_t, seq_lens_t, scale);
        } else {
            ops::self_attention_slots_decode(attn_out, q, kv_cache.k_cache, kv_cache.v_cache,
                                             slot_ids_t, seq_lens_t, scale);
        }

        tensor_t attn_out_flat = attn_out->view({batch_size, nh_per_rank_ * config_.dh});
        o_proj[rank] = meta.o_proj->slice(0, 0, batch_size);
        ops::linear(o_proj[rank], attn_out_flat, w.attn_o_w[layer_idx], nullptr);
    }

    allreduce_sum(o_proj);

    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        core::context().setDevice(device_type_, dev_id);
        auto& w = ranks_[rank].weights;
        auto& meta = decode_meta_[rank];

        hidden_states_1[rank] = meta.hidden_1->slice(0, 0, batch_size);
        ops::add(hidden_states_1[rank], hidden_states[rank], o_proj[rank]);

        tensor_t mlp_norm_out = meta.mlp_norm->slice(0, 0, batch_size);
        ops::rms_norm(mlp_norm_out, hidden_states_1[rank], w.mlp_norm_w[layer_idx], config_.epsilon);

        tensor_t gate_out = meta.gate_out->slice(0, 0, batch_size);
        tensor_t up_out = meta.up_out->slice(0, 0, batch_size);
        const char* fused_gate_env = std::getenv("LLAISYS_FUSED_GATE_UP");
        const bool use_fused_gate_up = fused_gate_env && std::atoi(fused_gate_env) != 0 &&
                                       w.mlp_gate_up_w[layer_idx];
        if (use_fused_gate_up) {
            tensor_t gate_up_out = meta.gate_up_out->slice(0, 0, batch_size);
            ops::linear(gate_up_out, mlp_norm_out, w.mlp_gate_up_w[layer_idx], nullptr);
            ops::split_gate_up_decode(gate_out, up_out, gate_up_out);
        } else {
            ops::linear(gate_out, mlp_norm_out, w.mlp_gate_w[layer_idx], nullptr);
            ops::linear(up_out, mlp_norm_out, w.mlp_up_w[layer_idx], nullptr);
        }

        tensor_t swiglu_out = meta.swiglu_out->slice(0, 0, batch_size);
        ops::swiglu(swiglu_out, gate_out, up_out);

        mlp_out[rank] = meta.mlp_out->slice(0, 0, batch_size);
        ops::linear(mlp_out[rank], swiglu_out, w.mlp_down_w[layer_idx], nullptr);
    }

    allreduce_sum(mlp_out);

    for (int rank = 0; rank < tp_size_; rank++) {
        int dev_id = device_ids_[rank];
        core::context().setDevice(device_type_, dev_id);
        auto& meta = decode_meta_[rank];

        tensor_t output;
        if (hidden_states[rank]->data() == meta.hidden_a->data()) {
            output = meta.hidden_b->slice(0, 0, batch_size);
        } else {
            output = meta.hidden_a->slice(0, 0, batch_size);
        }
        ops::add(output, hidden_states_1[rank], mlp_out[rank]);
        hidden_states[rank] = output;
    }
}

} // namespace models
} // namespace llaisys
