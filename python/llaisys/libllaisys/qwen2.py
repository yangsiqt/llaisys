"""
ctypes bindings for Qwen2 model C API
"""
from ctypes import POINTER, Structure, c_int, c_int64, c_float, c_size_t, c_void_p, c_char_p
from .tensor import llaisysTensor_t
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t


class LlaisysQwen2Meta(Structure):
    """C struct for Qwen2 model metadata"""
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


class LlaisysQwen2Weights(Structure):
    """C struct for Qwen2 model weights"""
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]


# Opaque pointer to Qwen2Model
llaisysQwen2Model_t = c_void_p


def load_qwen2(lib):
    """Load Qwen2 model functions from library"""
    
    # llaisysQwen2ModelCreate
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t
    
    # llaisysQwen2ModelDestroy
    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None
    
    # llaisysQwen2ModelWeights
    lib.llaisysQwen2ModelWeights.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    # Weight setters
    lib.llaisysQwen2ModelSetInEmbed.argtypes = [llaisysQwen2Model_t, llaisysTensor_t]
    lib.llaisysQwen2ModelSetInEmbed.restype = None

    lib.llaisysQwen2ModelSetOutEmbed.argtypes = [llaisysQwen2Model_t, llaisysTensor_t]
    lib.llaisysQwen2ModelSetOutEmbed.restype = None

    lib.llaisysQwen2ModelSetOutNormW.argtypes = [llaisysQwen2Model_t, llaisysTensor_t]
    lib.llaisysQwen2ModelSetOutNormW.restype = None

    lib.llaisysQwen2ModelSetLayerWeight.argtypes = [llaisysQwen2Model_t, c_char_p, c_size_t, llaisysTensor_t]
    lib.llaisysQwen2ModelSetLayerWeight.restype = None
    
    # llaisysQwen2ModelInfer
    lib.llaisysQwen2ModelInfer.argtypes = [
        llaisysQwen2Model_t,
        POINTER(c_int64),
        c_size_t,
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    # --- Tensor Parallel API ---
    llaisysQwen2TPModel_t = c_void_p

    lib.llaisysQwen2TPModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2TPModelCreate.restype = llaisysQwen2TPModel_t

    lib.llaisysQwen2TPModelDestroy.argtypes = [llaisysQwen2TPModel_t]
    lib.llaisysQwen2TPModelDestroy.restype = None

    lib.llaisysQwen2TPModelSetInEmbed.argtypes = [llaisysQwen2TPModel_t, c_int, llaisysTensor_t]
    lib.llaisysQwen2TPModelSetInEmbed.restype = None

    lib.llaisysQwen2TPModelSetOutEmbed.argtypes = [llaisysQwen2TPModel_t, llaisysTensor_t]
    lib.llaisysQwen2TPModelSetOutEmbed.restype = None

    lib.llaisysQwen2TPModelSetOutNormW.argtypes = [llaisysQwen2TPModel_t, c_int, llaisysTensor_t]
    lib.llaisysQwen2TPModelSetOutNormW.restype = None

    lib.llaisysQwen2TPModelSetLayerWeight.argtypes = [llaisysQwen2TPModel_t, c_int, c_char_p, c_size_t, llaisysTensor_t]
    lib.llaisysQwen2TPModelSetLayerWeight.restype = None

    lib.llaisysQwen2TPModelInfer.argtypes = [
        llaisysQwen2TPModel_t,
        POINTER(c_int64),
        c_size_t,
    ]
    lib.llaisysQwen2TPModelInfer.restype = c_int64

    lib.llaisysQwen2TPModelGetTPSize.argtypes = [llaisysQwen2TPModel_t]
    lib.llaisysQwen2TPModelGetTPSize.restype = c_int

