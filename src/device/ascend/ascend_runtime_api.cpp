#include "../runtime_api.hpp"

#include <acl/acl.h>

#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace llaisys::device::ascend {
namespace {

void checkRuntimeAcl(aclError status, const char *call) {
    if (status != ACL_SUCCESS) {
        std::ostringstream os;
        os << call << " failed with ACL error " << status;
        throw std::runtime_error(os.str());
    }
}

class AclLifetime {
public:
    AclLifetime() {
        aclError status = aclInit(nullptr);
        if (status == ACL_SUCCESS) owns_initialization_ = true;
        else if (status != ACL_ERROR_REPEAT_INITIALIZE) checkRuntimeAcl(status, "aclInit");
    }
    ~AclLifetime() {
        if (owns_initialization_) (void)aclFinalize();
    }

private:
    bool owns_initialization_ = false;
};

void ensureAclInitialized() {
    static AclLifetime lifetime;
    (void)lifetime;
}

aclrtMemcpyKind toAclMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H: return ACL_MEMCPY_HOST_TO_HOST;
    case LLAISYS_MEMCPY_H2D: return ACL_MEMCPY_HOST_TO_DEVICE;
    case LLAISYS_MEMCPY_D2H: return ACL_MEMCPY_DEVICE_TO_HOST;
    case LLAISYS_MEMCPY_D2D: return ACL_MEMCPY_DEVICE_TO_DEVICE;
    default: throw std::runtime_error("invalid LLAISYS memcpy kind");
    }
}

namespace runtime_api {

int getDeviceCount() {
    ensureAclInitialized();
    uint32_t count = 0;
    checkRuntimeAcl(aclrtGetDeviceCount(&count), "aclrtGetDeviceCount");
    return static_cast<int>(count);
}

void setDevice(int device_id) {
    ensureAclInitialized();
    checkRuntimeAcl(aclrtSetDevice(device_id), "aclrtSetDevice");
}

void deviceSynchronize() {
    checkRuntimeAcl(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice");
}

llaisysStream_t createStream() {
    aclrtStream stream = nullptr;
    checkRuntimeAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream) checkRuntimeAcl(aclrtDestroyStream(reinterpret_cast<aclrtStream>(stream)), "aclrtDestroyStream");
}

void streamSynchronize(llaisysStream_t stream) {
    checkRuntimeAcl(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream)), "aclrtSynchronizeStream");
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    checkRuntimeAcl(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr) checkRuntimeAcl(aclrtFree(ptr), "aclrtFree");
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    checkRuntimeAcl(aclrtMallocHost(&ptr, size), "aclrtMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr) checkRuntimeAcl(aclrtFreeHost(ptr), "aclrtFreeHost");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    if (kind == LLAISYS_MEMCPY_H2H) {
        std::memcpy(dst, src, size);
        return;
    }
    // Use and synchronize the default stream explicitly.  On CANN 9.0 the
    // nominally synchronous D2D entry point may not establish ordering with
    // a buffer produced by torch-npu until a later runtime operation.
    checkRuntimeAcl(aclrtMemcpyAsync(dst, size, src, size, toAclMemcpyKind(kind), nullptr),
                    "aclrtMemcpyAsync(sync)");
    checkRuntimeAcl(aclrtSynchronizeStream(nullptr), "aclrtSynchronizeStream(memcpy)");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    if (kind == LLAISYS_MEMCPY_H2H) {
        std::memcpy(dst, src, size);
        return;
    }
    checkRuntimeAcl(aclrtMemcpyAsync(dst, size, src, size, toAclMemcpyKind(kind),
                              reinterpret_cast<aclrtStream>(stream)),
             "aclrtMemcpyAsync");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount, &setDevice, &deviceSynchronize,
    &createStream, &destroyStream, &streamSynchronize,
    &mallocDevice, &freeDevice, &mallocHost, &freeHost,
    &memcpySync, &memcpyAsync};

} // namespace runtime_api
} // namespace

const LlaisysRuntimeAPI *getRuntimeAPI() {
    ensureAclInitialized();
    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::ascend
