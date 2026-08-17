add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

option("profiling")
    set_default(false)
    set_showmenu(true)
    set_description("Enable NVTX ranges and CUDA source line information")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    add_includedirs("/usr/local/cuda/targets/x86_64-linux/include")
    if has_config("profiling") then
        add_defines("LLAISYS_ENABLE_NVTX")
    end
    includes("xmake/nvidia.lua")
end

-- ASCEND --
option("ascend-npu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Ascend NPU")
option_end()

if has_config("ascend-npu") then
    add_defines("ENABLE_ASCEND_API")
    add_includedirs("/usr/local/Ascend/ascend-toolkit/latest/include")
    includes("xmake/ascend.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
    end
    if has_config("ascend-npu") then
        add_deps("llaisys-device-ascend")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia")
    end
    if has_config("ascend-npu") then
        add_deps("llaisys-ops-ascend")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-models")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/models/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    add_deps("llaisys-models")
    if has_config("nv-gpu") then
        add_rules("cuda")
        add_cugencodes("sm_86")
        add_cuflags("-Xcompiler=-fPIC", {force = true})
        if has_config("profiling") then
            add_cuflags("-lineinfo", {force = true})
        end
        add_files("src/llaisys/cuda_devlink_stub.cu")
        add_links("cudart", "cublas")
        add_linkdirs("/usr/local/cuda/lib64")
    end
    if has_config("ascend-npu") then
        add_links("ascendcl", "opapi", "nnopbase", "atb")
        add_linkdirs("/usr/local/Ascend/ascend-toolkit/latest/lib64")
        add_linkdirs("/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib")
        add_rpathdirs("/usr/local/Ascend/ascend-toolkit/latest/lib64")
        add_rpathdirs("/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()
