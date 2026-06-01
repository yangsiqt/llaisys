#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);
    
    // Set weights (helper functions)
    __export void llaisysQwen2ModelSetInEmbed(struct LlaisysQwen2Model * model, llaisysTensor_t tensor);
    __export void llaisysQwen2ModelSetOutEmbed(struct LlaisysQwen2Model * model, llaisysTensor_t tensor);
    __export void llaisysQwen2ModelSetOutNormW(struct LlaisysQwen2Model * model, llaisysTensor_t tensor);
    __export void llaisysQwen2ModelSetLayerWeight(struct LlaisysQwen2Model * model, const char* name, size_t layer_idx, llaisysTensor_t tensor);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);

    // --- Tensor Parallel API ---
    struct LlaisysQwen2TPModel;

    __export struct LlaisysQwen2TPModel *llaisysQwen2TPModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);
    __export void llaisysQwen2TPModelDestroy(struct LlaisysQwen2TPModel *model);

    __export void llaisysQwen2TPModelSetInEmbed(struct LlaisysQwen2TPModel *model, int rank, llaisysTensor_t tensor);
    __export void llaisysQwen2TPModelSetOutEmbed(struct LlaisysQwen2TPModel *model, llaisysTensor_t tensor);
    __export void llaisysQwen2TPModelSetOutNormW(struct LlaisysQwen2TPModel *model, int rank, llaisysTensor_t tensor);
    __export void llaisysQwen2TPModelSetLayerWeight(struct LlaisysQwen2TPModel *model, int rank, const char *name, size_t layer_idx, llaisysTensor_t tensor);

    __export int64_t llaisysQwen2TPModelInfer(struct LlaisysQwen2TPModel *model, int64_t *token_ids, size_t ntoken);
    __export int llaisysQwen2TPModelInferBatch(struct LlaisysQwen2TPModel *model, int64_t *token_ids, size_t ntoken, int64_t *out, size_t batch_size);
    __export int llaisysQwen2TPModelInitContinuous(struct LlaisysQwen2TPModel *model, size_t max_slots);
    __export int64_t llaisysQwen2TPModelPrefillSlot(struct LlaisysQwen2TPModel *model, size_t slot_id, int64_t *token_ids, size_t ntoken);
    __export int llaisysQwen2TPModelPrefillSlotChunk(struct LlaisysQwen2TPModel *model, size_t slot_id, int64_t *token_ids, size_t ntoken, int final_chunk, int64_t *out);
    __export int llaisysQwen2TPModelPrefillSlots(struct LlaisysQwen2TPModel *model, size_t *slot_ids, int64_t *token_ids, int64_t *out, size_t nslot, size_t prompt_len);
    __export int llaisysQwen2TPModelDecodeSlots(struct LlaisysQwen2TPModel *model, size_t *slot_ids, int64_t *input_tokens, int64_t *out, size_t nslot);
    __export int llaisysQwen2TPModelReleaseSlot(struct LlaisysQwen2TPModel *model, size_t slot_id);
    __export size_t llaisysQwen2TPModelSlotSeqLen(struct LlaisysQwen2TPModel *model, size_t slot_id);
    __export int llaisysQwen2TPModelGetTPSize(struct LlaisysQwen2TPModel *model);
}
#endif // LLAISYS_MODELS_QWEN2_H
