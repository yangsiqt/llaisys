target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", "-fopenmp", "-mavx512f", "-mavx512bw", "-mavx512vl", "-mfma", "-O3")
        add_defines("LLAISYS_USE_OPENBLAS")
        add_links("openblas", "gomp")
        add_includedirs("/usr/include/x86_64-linux-gnu")
        add_linkdirs("/usr/lib/x86_64-linux-gnu")
    end

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()

