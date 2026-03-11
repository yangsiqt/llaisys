#ifndef LLAISYS_MODELS_QWEN2_TP_HPP
#define LLAISYS_MODELS_QWEN2_TP_HPP

#include "qwen2.hpp"

#ifdef ENABLE_NVIDIA_API
#include "../../device/nvidia/nccl_comm.h"
#endif

namespace llaisys {
namespace models {

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
    void reset_cache();

private:
    Qwen2Config config_;
    llaisysDeviceType_t device_type_;
    std::vector<int> device_ids_;
    int tp_size_;

    size_t nh_per_rank_;
    size_t nkvh_per_rank_;
    size_t di_per_rank_;

    struct RankData {
        Qwen2Weights weights;
        std::vector<KVCache> kv_caches;
    };
    std::vector<RankData> ranks_;

    size_t current_pos_;

#ifdef ENABLE_NVIDIA_API
    std::unique_ptr<device::nvidia::NcclComm> nccl_comm_;
#endif

    tensor_t forward(const std::vector<int64_t>& new_tokens, size_t start_pos, size_t seq_len);
    void apply_layer(size_t layer_idx, std::vector<tensor_t>& hidden_states,
                     size_t start_pos, size_t seq_len);
    void allreduce_sum(std::vector<tensor_t>& tensors);
};

} // namespace models
} // namespace llaisys

#endif // LLAISYS_MODELS_QWEN2_TP_HPP
