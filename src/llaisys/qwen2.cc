#include "llaisys_tensor.hpp"
#include "../models/qwen2/qwen2.hpp"
#include "llaisys/models/qwen2.h"
#include <cstring>

using namespace llaisys;
using namespace llaisys::models;

extern "C" {

struct LlaisysQwen2Model {
    Qwen2Model *model;
};

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {
    
    // Convert C config to C++ config
    Qwen2Config config;
    config.dtype = meta->dtype;
    config.nlayer = meta->nlayer;
    config.hs = meta->hs;
    config.nh = meta->nh;
    config.nkvh = meta->nkvh;
    config.dh = meta->dh;
    config.di = meta->di;
    config.maxseq = meta->maxseq;
    config.voc = meta->voc;
    config.epsilon = meta->epsilon;
    config.theta = meta->theta;
    config.end_token = meta->end_token;
    
    // Convert device IDs
    std::vector<int> dev_ids;
    for (int i = 0; i < ndevice; i++) {
        dev_ids.push_back(device_ids[i]);
    }
    
    // Create model
    auto *llaisys_model = new LlaisysQwen2Model();
    llaisys_model->model = new Qwen2Model(config, device, dev_ids);
    
    return llaisys_model;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (model) {
        delete model->model;
        delete model;
    }
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (!model) return nullptr;

    const Qwen2Config &config = model->model->config();

    // Allocate C-style weights structure (used for introspection only).
    // Note: Python should NOT rely on mutating this struct to set model weights.
    auto *c_weights = new LlaisysQwen2Weights();
    std::memset(c_weights, 0, sizeof(LlaisysQwen2Weights));

    c_weights->attn_norm_w = new llaisysTensor_t[config.nlayer];
    c_weights->attn_q_w = new llaisysTensor_t[config.nlayer];
    c_weights->attn_q_b = new llaisysTensor_t[config.nlayer];
    c_weights->attn_k_w = new llaisysTensor_t[config.nlayer];
    c_weights->attn_k_b = new llaisysTensor_t[config.nlayer];
    c_weights->attn_v_w = new llaisysTensor_t[config.nlayer];
    c_weights->attn_v_b = new llaisysTensor_t[config.nlayer];
    c_weights->attn_o_w = new llaisysTensor_t[config.nlayer];
    c_weights->mlp_norm_w = new llaisysTensor_t[config.nlayer];
    c_weights->mlp_gate_w = new llaisysTensor_t[config.nlayer];
    c_weights->mlp_up_w = new llaisysTensor_t[config.nlayer];
    c_weights->mlp_down_w = new llaisysTensor_t[config.nlayer];

    for (size_t i = 0; i < config.nlayer; i++) {
        c_weights->attn_norm_w[i] = nullptr;
        c_weights->attn_q_w[i] = nullptr;
        c_weights->attn_q_b[i] = nullptr;
        c_weights->attn_k_w[i] = nullptr;
        c_weights->attn_k_b[i] = nullptr;
        c_weights->attn_v_w[i] = nullptr;
        c_weights->attn_v_b[i] = nullptr;
        c_weights->attn_o_w[i] = nullptr;
        c_weights->mlp_norm_w[i] = nullptr;
        c_weights->mlp_gate_w[i] = nullptr;
        c_weights->mlp_up_w[i] = nullptr;
        c_weights->mlp_down_w[i] = nullptr;
    }

    return c_weights;
}

// Helper function to set weight in C++ model
void llaisysQwen2ModelSetInEmbed(struct LlaisysQwen2Model *model, llaisysTensor_t tensor) {
    if (!model || !model->model || !tensor) return;
    auto *t = reinterpret_cast<LlaisysTensor *>(tensor);
    model->model->weights().in_embed = t->tensor;
}

void llaisysQwen2ModelSetOutEmbed(struct LlaisysQwen2Model *model, llaisysTensor_t tensor) {
    if (!model || !model->model || !tensor) return;
    auto *t = reinterpret_cast<LlaisysTensor *>(tensor);
    model->model->weights().out_embed = t->tensor;
}

void llaisysQwen2ModelSetOutNormW(struct LlaisysQwen2Model *model, llaisysTensor_t tensor) {
    if (!model || !model->model || !tensor) return;
    auto *t = reinterpret_cast<LlaisysTensor *>(tensor);
    model->model->weights().out_norm_w = t->tensor;
}

void llaisysQwen2ModelSetLayerWeight(struct LlaisysQwen2Model *model, const char* name, size_t layer_idx, llaisysTensor_t tensor) {
    if (!model || !model->model || !tensor) return;
    auto *t = reinterpret_cast<LlaisysTensor *>(tensor);
    tensor_t t_shared = t->tensor;

    const Qwen2Config &config = model->model->config();
    Qwen2Weights &w = model->model->weights();

    // Ensure vectors are sized.
    if (w.attn_norm_w.size() != config.nlayer) {
        w.attn_norm_w.resize(config.nlayer);
        w.attn_q_w.resize(config.nlayer);
        w.attn_q_b.resize(config.nlayer);
        w.attn_k_w.resize(config.nlayer);
        w.attn_k_b.resize(config.nlayer);
        w.attn_v_w.resize(config.nlayer);
        w.attn_v_b.resize(config.nlayer);
        w.attn_o_w.resize(config.nlayer);
        w.mlp_norm_w.resize(config.nlayer);
        w.mlp_gate_w.resize(config.nlayer);
        w.mlp_up_w.resize(config.nlayer);
        w.mlp_down_w.resize(config.nlayer);
    }
    if (layer_idx >= config.nlayer) return;

    std::string weight_name(name ? name : "");

    if (weight_name == "attn_norm_w")
        w.attn_norm_w[layer_idx] = t_shared;
    else if (weight_name == "attn_q_w")
        w.attn_q_w[layer_idx] = t_shared;
    else if (weight_name == "attn_q_b")
        w.attn_q_b[layer_idx] = t_shared;
    else if (weight_name == "attn_k_w")
        w.attn_k_w[layer_idx] = t_shared;
    else if (weight_name == "attn_k_b")
        w.attn_k_b[layer_idx] = t_shared;
    else if (weight_name == "attn_v_w")
        w.attn_v_w[layer_idx] = t_shared;
    else if (weight_name == "attn_v_b")
        w.attn_v_b[layer_idx] = t_shared;
    else if (weight_name == "attn_o_w")
        w.attn_o_w[layer_idx] = t_shared;
    else if (weight_name == "mlp_norm_w")
        w.mlp_norm_w[layer_idx] = t_shared;
    else if (weight_name == "mlp_gate_w")
        w.mlp_gate_w[layer_idx] = t_shared;
    else if (weight_name == "mlp_up_w")
        w.mlp_up_w[layer_idx] = t_shared;
    else if (weight_name == "mlp_down_w")
        w.mlp_down_w[layer_idx] = t_shared;
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (!model || !model->model) return -1;
    
    std::vector<int64_t> tokens(token_ids, token_ids + ntoken);
    return model->model->infer(tokens);
}

} // extern "C"
