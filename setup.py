# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

extra_cuda_cflags = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    # add your arch here if you like; otherwise set TORCH_CUDA_ARCH_LIST env
    # e.g. "-gencode=arch=compute_90,code=sm_90",
]
extra_cflags = ["-O3", "-std=c++17"]

ext_modules = []

# --- QK attention scores (your existing qk_attn_ext) ---
ext_modules.append(
    CUDAExtension(
        name="qk_attn_ext",
        sources=[
            "ext/qk_attn.cpp",
            "ext/qk_attn_kernel.cu",
        ],
        extra_compile_args={
            "cxx": extra_cflags,
            "nvcc": extra_cuda_cflags,
        },
    )
)

# --- Extra kernels: AV / RMSNorm / RoPE (your attn_extra_ext) ---
ext_modules.append(
    CUDAExtension(
        name="attn_extra_ext",
        sources=[
            "ext/attn_extra.cpp",   # C++ pybind for the extra kernels
            "ext/av_attn.cu",       # A·V kernel
            "ext/rmsnorm.cu",       # RMSNorm kernel
            "ext/rope_apply.cu",    # RoPE apply kernel
        ],
        extra_compile_args={
            "cxx": extra_cflags,
            "nvcc": extra_cuda_cflags,
        },
    )
)

setup(
    name="mistral_custom_kernels",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
