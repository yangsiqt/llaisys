#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// Self-Attention with causal mask and GQA (Grouped Query Attention), batch supported.
// Q: [qlen, n_heads, head_dim]   — flattened batch (total queries = num_seqs * qlen_per_seq)
// K: [kvlen, n_kv_heads, head_dim] — flattened batch (total keys = num_seqs * kvlen_per_seq)
// V: [kvlen, n_kv_heads, head_dim]
// Output: [qlen, n_heads, head_dim]

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale,
                     size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads, size_t head_dim,
                     size_t qlen_per_seq) {

    size_t n_rep = n_heads / n_kv_heads;
    size_t num_seqs = qlen / qlen_per_seq;
    size_t kvlen_per_seq = kvlen / num_seqs;

    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        std::vector<float> scores(qlen * kvlen);

        for (size_t h = 0; h < n_heads; h++) {
            size_t kv_h = h / n_rep;

            for (size_t qi = 0; qi < qlen; qi++) {
                size_t seq_id = qi / qlen_per_seq;
                size_t local_qi = qi % qlen_per_seq;
                size_t abs_local_qi = kvlen_per_seq - qlen_per_seq + local_qi;

                for (size_t ki = 0; ki < kvlen; ki++) {
                    size_t ki_seq_id = ki / kvlen_per_seq;
                    size_t local_ki = ki % kvlen_per_seq;

                    float score;
                    if (ki_seq_id != seq_id || local_ki > abs_local_qi) {
                        score = -std::numeric_limits<float>::infinity();
                    } else {
                        score = 0.0f;
                        for (size_t d = 0; d < head_dim; d++) {
                            float q_val = llaisys::utils::cast<float>(q[qi * n_heads * head_dim + h * head_dim + d]);
                            float k_val = llaisys::utils::cast<float>(k[ki * n_kv_heads * head_dim + kv_h * head_dim + d]);
                            score += q_val * k_val;
                        }
                        score *= scale;
                    }

                    scores[qi * kvlen + ki] = score;
                }

                // Softmax over kvlen dimension
                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t ki = 0; ki < kvlen; ki++)
                    max_score = std::max(max_score, scores[qi * kvlen + ki]);

                float sum_exp = 0.0f;
                for (size_t ki = 0; ki < kvlen; ki++) {
                    scores[qi * kvlen + ki] = std::exp(scores[qi * kvlen + ki] - max_score);
                    sum_exp += scores[qi * kvlen + ki];
                }

                for (size_t ki = 0; ki < kvlen; ki++)
                    scores[qi * kvlen + ki] /= sum_exp;

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
        std::vector<float> scores(qlen * kvlen);

        for (size_t h = 0; h < n_heads; h++) {
            size_t kv_h = h / n_rep;

            for (size_t qi = 0; qi < qlen; qi++) {
                size_t seq_id = qi / qlen_per_seq;
                size_t local_qi = qi % qlen_per_seq;
                size_t abs_local_qi = kvlen_per_seq - qlen_per_seq + local_qi;

                for (size_t ki = 0; ki < kvlen; ki++) {
                    size_t ki_seq_id = ki / kvlen_per_seq;
                    size_t local_ki = ki % kvlen_per_seq;

                    float score;
                    if (ki_seq_id != seq_id || local_ki > abs_local_qi) {
                        score = -std::numeric_limits<float>::infinity();
                    } else {
                        score = 0.0f;
                        for (size_t d = 0; d < head_dim; d++) {
                            float q_val = static_cast<float>(q[qi * n_heads * head_dim + h * head_dim + d]);
                            float k_val = static_cast<float>(k[ki * n_kv_heads * head_dim + kv_h * head_dim + d]);
                            score += q_val * k_val;
                        }
                        score *= scale;
                    }

                    scores[qi * kvlen + ki] = score;
                }

                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t ki = 0; ki < kvlen; ki++)
                    max_score = std::max(max_score, scores[qi * kvlen + ki]);

                float sum_exp = 0.0f;
                for (size_t ki = 0; ki < kvlen; ki++) {
                    scores[qi * kvlen + ki] = std::exp(scores[qi * kvlen + ki] - max_score);
                    sum_exp += scores[qi * kvlen + ki];
                }

                for (size_t ki = 0; ki < kvlen; ki++)
                    scores[qi * kvlen + ki] /= sum_exp;

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
                    size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads, size_t head_dim,
                    size_t qlen_per_seq) {

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                              reinterpret_cast<const float *>(q),
                              reinterpret_cast<const float *>(k),
                              reinterpret_cast<const float *>(v),
                              scale, qlen, kvlen, n_heads, n_kv_heads, head_dim, qlen_per_seq);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                              reinterpret_cast<const llaisys::bf16_t *>(q),
                              reinterpret_cast<const llaisys::bf16_t *>(k),
                              reinterpret_cast<const llaisys::bf16_t *>(v),
                              scale, qlen, kvlen, n_heads, n_kv_heads, head_dim, qlen_per_seq);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                              reinterpret_cast<const llaisys::fp16_t *>(q),
                              reinterpret_cast<const llaisys::fp16_t *>(k),
                              reinterpret_cast<const llaisys::fp16_t *>(v),
                              scale, qlen, kvlen, n_heads, n_kv_heads, head_dim, qlen_per_seq);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
