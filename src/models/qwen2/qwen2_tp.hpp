#ifndef LLAISYS_MODELS_QWEN2_TP_HPP
#define LLAISYS_MODELS_QWEN2_TP_HPP

#include "qwen2.hpp"

#ifdef ENABLE_NVIDIA_API
#include "../../device/nvidia/nccl_comm.h"
#endif

namespace llaisys {
namespace models {

struct Qwen2TPRankData {
    Qwen2Weights weights;
    std::vector<KVCache> kv_caches;
    std::vector<KVCache> paged_kv_caches;
};

struct PagedKVStats {
    size_t block_size;
    size_t max_blocks;
    size_t used_blocks;
    size_t peak_used_blocks;
    size_t free_blocks;
    size_t kv_capacity_tokens;
};

class Qwen2TPModel {
public:
    Qwen2TPModel(const Qwen2Config& config, llaisysDeviceType_t device_type,
                 const std::vector<int>& device_ids);
    ~Qwen2TPModel();

    const Qwen2Config& config() const { return config_; }
    int tp_size() const { return tp_size_; }

    void setInEmbed(int rank, const tensor_t& tensor);
    void setOutEmbed(const tensor_t& tensor);
    void setOutNormW(int rank, const tensor_t& tensor);
    void setLayerWeight(int rank, const std::string& name, size_t layer_idx, const tensor_t& tensor);

    int64_t infer(const std::vector<int64_t>& token_ids);
    std::vector<int64_t> infer_batch(const std::vector<int64_t>& token_ids, size_t batch_size);
    void reset_cache();

    void init_continuous(size_t max_slots);
    void init_paged_continuous(size_t max_slots, size_t block_size,
                               size_t max_blocks, size_t prefill_scratch_slots);
    int64_t prefill_slot(size_t slot_id, const std::vector<int64_t>& token_ids);
    int64_t prefill_slot_chunk(size_t slot_id, const std::vector<int64_t>& token_ids, bool final_chunk);
    std::vector<int64_t> prefill_slots(const std::vector<size_t>& slot_ids,
                                       const std::vector<int64_t>& token_ids,
                                       size_t prompt_len);
    std::vector<int64_t> prefill_slots_varlen(const std::vector<size_t>& slot_ids,
                                              const std::vector<int64_t>& token_ids,
                                              const std::vector<size_t>& prompt_lens,
                                              size_t max_prompt_len);
    std::vector<int64_t> decode_slots(const std::vector<size_t>& slot_ids,
                                      const std::vector<int64_t>& input_tokens);
    void release_slot(size_t slot_id);
    size_t slot_seq_len(size_t slot_id) const;
    PagedKVStats paged_kv_stats() const;

private:
    Qwen2Config config_;
    llaisysDeviceType_t device_type_;
    std::vector<int> device_ids_;
    int tp_size_;

    size_t nh_per_rank_;
    size_t nkvh_per_rank_;
    size_t di_per_rank_;

    std::vector<Qwen2TPRankData> ranks_;

    size_t current_pos_;
    size_t batch_size_;
    size_t max_slots_;
    bool paged_kv_mode_;
    size_t paged_block_size_;
    size_t paged_max_blocks_;
    size_t paged_max_blocks_per_slot_;
    size_t prefill_scratch_slots_;
    size_t paged_peak_used_blocks_;
    bool continuous_ready_;
    std::vector<uint8_t> slot_active_;
    std::vector<size_t> slot_seq_lens_;
    std::vector<int64_t> slot_last_tokens_;
    std::vector<std::vector<int64_t>> slot_block_tables_;
    std::vector<size_t> free_paged_blocks_;
    struct DecodeMetaBuffers {
        tensor_t slot_ids;
        tensor_t positions;
        tensor_t seq_lens;
        tensor_t pos_ids;
        tensor_t max_idx;
        tensor_t max_val;
        tensor_t partial_max_idx;
        tensor_t partial_max_val;
        tensor_t input_ids;
        tensor_t hidden_a;
        tensor_t hidden_b;
        tensor_t attn_norm;
        tensor_t qkv_flat;
        tensor_t q_flat;
        tensor_t k_flat;
        tensor_t v_flat;
        tensor_t attn_out;
        tensor_t o_proj;
        tensor_t hidden_1;
        tensor_t mlp_norm;
        tensor_t gate_up_out;
        tensor_t gate_out;
        tensor_t up_out;
        tensor_t swiglu_out;
        tensor_t mlp_out;
        tensor_t out_norm;
        tensor_t logits;
        tensor_t block_tables;
    };
    std::vector<DecodeMetaBuffers> decode_meta_;
    struct PrefillMetaBuffers {
        tensor_t input_ids;
        tensor_t hidden_a;
        tensor_t hidden_b;
        tensor_t attn_norm;
        tensor_t qkv_flat;
        tensor_t q_flat;
        tensor_t k_flat;
        tensor_t v_flat;
        tensor_t pos_ids;
        tensor_t attn_out;
        tensor_t o_proj;
        tensor_t hidden_1;
        tensor_t mlp_norm;
        tensor_t gate_up_out;
        tensor_t gate_out;
        tensor_t up_out;
        tensor_t swiglu_out;
        tensor_t mlp_out;
        tensor_t out_norm;
        tensor_t logits;
    };
    std::vector<PrefillMetaBuffers> prefill_meta_;
    size_t prefill_workspace_capacity_;

#ifdef ENABLE_NVIDIA_API
    std::unique_ptr<device::nvidia::NcclComm> nccl_comm_;
    bool decode_graph_ready_;
    size_t decode_graph_batch_;
    void* decode_graph_;
    void* decode_graph_exec_;
#endif

    tensor_t forward(const std::vector<int64_t>& new_tokens, size_t start_pos, size_t seq_len, size_t batch_size);
    tensor_t forward_slots(const std::vector<int64_t>& new_tokens, const std::vector<size_t>& slot_ids,
                           size_t start_pos, size_t seq_len);
    tensor_t forward_slots_decode(const std::vector<int64_t>& new_tokens, const std::vector<size_t>& slot_ids,
                                  const std::vector<size_t>& start_positions);
    tensor_t forward_slots_decode_compute(size_t batch_size, const std::vector<size_t>& slot_ids,
                                          const std::vector<size_t>& start_positions);
    void apply_layer(size_t layer_idx, std::vector<tensor_t>& hidden_states,
                     size_t start_pos, size_t seq_len, size_t batch_size);
    void apply_layer_slots(size_t layer_idx, std::vector<tensor_t>& hidden_states,
                           const std::vector<size_t>& slot_ids, size_t start_pos, size_t seq_len);
    void apply_layer_slots_decode(size_t layer_idx, std::vector<tensor_t>& hidden_states,
                                  const std::vector<size_t>& slot_ids,
                                  const std::vector<size_t>& start_positions);
    void allreduce_sum(std::vector<tensor_t>& tensors, const char* label = nullptr);
    void init_continuous_common(size_t max_slots, size_t kv_slots);
    void ensure_prefill_workspace(size_t flat_seq_len);
    void reset_paged_blocks();
    void release_paged_blocks(size_t slot_id);
    void ensure_paged_position_allocated(size_t slot_id, size_t position);
    void allocate_paged_blocks_for_len(size_t slot_id, size_t seq_len);
    void sync_paged_block_table(int rank);
    void copy_prefill_to_paged_cache(const std::vector<size_t>& real_slot_ids,
                                     const std::vector<size_t>& scratch_slot_ids,
                                     size_t prompt_len);
};

} // namespace models
} // namespace llaisys

#endif // LLAISYS_MODELS_QWEN2_TP_HPP
