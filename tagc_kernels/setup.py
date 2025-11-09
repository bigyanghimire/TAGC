from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tagc_kernels',
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            'tagc_kernels',
            ['kernels/tagc.cpp', 'kernels/tagc_kernels.cu'],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
