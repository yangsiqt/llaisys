#ifndef LLAISYS_UTILS_NVTX_HPP
#define LLAISYS_UTILS_NVTX_HPP

#ifdef LLAISYS_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace llaisys::utils {

class NvtxRange {
public:
    explicit NvtxRange(const char *name) {
#ifdef LLAISYS_ENABLE_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }

    ~NvtxRange() {
#ifdef LLAISYS_ENABLE_NVTX
        nvtxRangePop();
#endif
    }

    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;
};

} // namespace llaisys::utils

#define LLAISYS_NVTX_CONCAT_INNER(a, b) a##b
#define LLAISYS_NVTX_CONCAT(a, b) LLAISYS_NVTX_CONCAT_INNER(a, b)
#define LLAISYS_NVTX_RANGE(name) \
    ::llaisys::utils::NvtxRange LLAISYS_NVTX_CONCAT(_llaisys_nvtx_range_, __LINE__)(name)

#endif // LLAISYS_UTILS_NVTX_HPP
