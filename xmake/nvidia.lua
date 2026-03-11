local nccl_inc = "/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/include"
local nccl_lib = "/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib"
if not os.isdir(nccl_inc) then
    nccl_inc = "/usr/include"
    nccl_lib = "/usr/lib/x86_64-linux-gnu"
end

target("llaisys-device-nvidia")
    set_kind("static")
    add_rules("cuda")
    set_languages("cxx17")
    add_cugencodes("sm_86")
    add_cuflags("-Xcompiler=-fPIC", {force = true})

    add_includedirs(nccl_inc)
    add_files("../src/device/nvidia/nvidia_runtime_api.cu")
    add_files("../src/device/nvidia/nvidia_resource.cu")
    add_files("../src/device/nvidia/nccl_comm.cu")
    add_linkdirs(nccl_lib)
    add_links("nccl")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_rules("cuda")
    set_languages("cxx17")
    add_cugencodes("sm_86")
    add_cuflags("-Xcompiler=-fPIC", {force = true})
    add_cuflags("--expt-relaxed-constexpr", {force = true})

    add_includedirs("../third_party/cutlass/include")
    add_links("cublas")
    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
