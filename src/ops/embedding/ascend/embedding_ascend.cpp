#include "embedding_ascend.hpp"
#include "../../../device/ascend/aclnn_utils.hpp"
#include <aclnnop/aclnn_embedding.h>
namespace llaisys::ops::ascend {
void embedding(void *out, const void *index, const void *weight, llaisysDataType_t dtype, size_t seq_len, size_t hidden_size, size_t vocab_size) {
    device::ascend::AclTensorDesc w(const_cast<void *>(weight), dtype, {static_cast<int64_t>(vocab_size), static_cast<int64_t>(hidden_size)});
    device::ascend::AclTensorDesc idx(const_cast<void *>(index), LLAISYS_DTYPE_I64, {static_cast<int64_t>(seq_len)});
    device::ascend::AclTensorDesc o(out, dtype, {static_cast<int64_t>(seq_len), static_cast<int64_t>(hidden_size)});
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    device::ascend::checkAclnn(aclnnEmbeddingGetWorkspaceSize(w.get(), idx.get(), o.get(), &ws, &ex), "aclnnEmbeddingGetWorkspaceSize");
    device::ascend::runAclnn(ws, ex, "aclnnEmbedding", aclnnEmbedding);
}
}
