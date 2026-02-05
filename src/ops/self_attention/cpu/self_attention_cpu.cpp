#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// Self-Attention with causal mask and GQA (Grouped Query Attention)
// Q: [qlen, n_heads, head_dim]
// K: [kvlen, n_kv_heads, head_dim]
// V: [kvlen, n_kv_heads, head_dim]
// Output: [qlen, n_heads, head_dim]
// 
// Steps:
// 1. Compute attention scores: A = Q @ K^T * scale
// 2. Apply causal mask (for positions > kvlen - qlen from current query position)
// 3. Apply softmax
// 4. Compute attention values: attn_val = softmax(A) @ V
// Note: K and V are repeated if n_heads > n_kv_heads (GQA)

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale,
                     size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    
    size_t n_rep = n_heads / n_kv_heads;  // repetition factor for GQA
    
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        // For half precision, use float for computation
        std::vector<float> scores(qlen * kvlen);
        
        for (size_t h = 0; h < n_heads; h++) {
            size_t kv_h = h / n_rep;  // which kv head to use
            
            // Compute attention scores: Q[qlen, head_dim] @ K^T[head_dim, kvlen]
            for (size_t qi = 0; qi < qlen; qi++) {
                for (size_t ki = 0; ki < kvlen; ki++) {
                    float score = 0.0f;
                    for (size_t d = 0; d < head_dim; d++) {
                        float q_val = llaisys::utils::cast<float>(q[qi * n_heads * head_dim + h * head_dim + d]);
                        float k_val = llaisys::utils::cast<float>(k[ki * n_kv_heads * head_dim + kv_h * head_dim + d]);
                        score += q_val * k_val;
                    }
                    score *= scale;
                    
                    // Apply causal mask: can only attend to positions <= current position in the full sequence
                    // kvlen is the total context length, qlen is the query length
                    // Query position qi corresponds to absolute position (kvlen - qlen + qi)
                    size_t abs_qi = kvlen - qlen + qi;
                    if (ki > abs_qi) {
                        score = -std::numeric_limits<float>::infinity();
                    }
                    
                    scores[qi * kvlen + ki] = score;
                }
                
                // Softmax over kvlen dimension
                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t ki = 0; ki < kvlen; ki++) {
                    max_score = std::max(max_score, scores[qi * kvlen + ki]);
                }
                
                float sum_exp = 0.0f;
                for (size_t ki = 0; ki < kvlen; ki++) {
                    scores[qi * kvlen + ki] = std::exp(scores[qi * kvlen + ki] - max_score);
                    sum_exp += scores[qi * kvlen + ki];
                }
                
                for (size_t ki = 0; ki < kvlen; ki++) {
                    scores[qi * kvlen + ki] /= sum_exp;
                }
                
                // Compute output: softmax(scores) @ V
                for (size_t d = 0; d < head_dim; d++) {
                    float val = 0.0f;
                    for (size_t ki = 0; ki < kvlen; ki++) {
                        float v_val = llaisys::utils::cast<float>(v[ki * n_kv_heads * head_dim + kv_h * head_dim + d]);
                        val += scores[qi * kvlen + ki] * v_val;
                    }
                    attn_val[qi * n_heads * head_dim + h * head_dim + d] = llaisys::utils::cast<T>(val);
                }
            }
        }
    } else {
        // For full precision
        std::vector<float> scores(qlen * kvlen);
        
        for (size_t h = 0; h < n_heads; h++) {
            size_t kv_h = h / n_rep;  // which kv head to use
            
            // Compute attention scores
            for (size_t qi = 0; qi < qlen; qi++) {
                for (size_t ki = 0; ki < kvlen; ki++) {
                    float score = 0.0f;
                    for (size_t d = 0; d < head_dim; d++) {
                        float q_val = static_cast<float>(q[qi * n_heads * head_dim + h * head_dim + d]);
                        float k_val = static_cast<float>(k[ki * n_kv_heads * head_dim + kv_h * head_dim + d]);
                        score += q_val * k_val;
                    }
                    score *= scale;
                    
                    // Apply causal mask
                    size_t abs_qi = kvlen - qlen + qi;
                    if (ki > abs_qi) {
                        score = -std::numeric_limits<float>::infinity();
                    }
                    
                    scores[qi * kvlen + ki] = score;
                }
                
                // Softmax
                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t ki = 0; ki < kvlen; ki++) {
                    max_score = std::max(max_score, scores[qi * kvlen + ki]);
                }
                
                float sum_exp = 0.0f;
                for (size_t ki = 0; ki < kvlen; ki++) {
                    scores[qi * kvlen + ki] = std::exp(scores[qi * kvlen + ki] - max_score);
                    sum_exp += scores[qi * kvlen + ki];
                }
                
                for (size_t ki = 0; ki < kvlen; ki++) {
                    scores[qi * kvlen + ki] /= sum_exp;
                }
                
                // Compute output
                for (size_t d = 0; d < head_dim; d++) {
                    float val = 0.0f;
                    for (size_t ki = 0; ki < kvlen; ki++) {
                        float v_val = static_cast<float>(v[ki * n_kv_heads * head_dim + kv_h * head_dim + d]);
                        val += scores[qi * kvlen + ki] * v_val;
                    }
                    attn_val[qi * n_heads * head_dim + h * head_dim + d] = static_cast<T>(val);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, 
                    float scale, llaisysDataType_t type, 
                    size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                              reinterpret_cast<const float *>(q),
                              reinterpret_cast<const float *>(k),
                              reinterpret_cast<const float *>(v),
                              scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                              reinterpret_cast<const llaisys::bf16_t *>(q),
                              reinterpret_cast<const llaisys::bf16_t *>(k),
                              reinterpret_cast<const llaisys::bf16_t *>(v),
                              scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                              reinterpret_cast<const llaisys::fp16_t *>(q),
                              reinterpret_cast<const llaisys::fp16_t *>(k),
                              reinterpret_cast<const llaisys::fp16_t *>(v),
                              scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu

