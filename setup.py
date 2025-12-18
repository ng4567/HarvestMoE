from setuptools import setup, find_packages, Extension
import os
import sys
import subprocess
import torch
import torch.utils.cpp_extension as torch_cpp_ext
import cpuinfo

def get_mkl_include_path():
    """Find MKL include path from various sources."""
    # Try mkl_include package first
    try:
        import mkl_include
        return mkl_include.get_include()
    except ImportError:
        pass
    
    # Try the venv's include directory directly (where mkl package installs headers)
    venv_include = os.path.join(sys.prefix, 'include')
    if os.path.exists(os.path.join(venv_include, 'mkl.h')):
        return venv_include
    
    # Try to find in venv's site-packages
    for path in sys.path:
        mkl_path = os.path.join(path, 'mkl_include', 'include')
        if os.path.exists(mkl_path):
            return mkl_path
    
    # Try Intel oneAPI standard location
    oneapi_path = '/opt/intel/oneapi/mkl/latest/include'
    if os.path.exists(oneapi_path):
        return oneapi_path
    
    # Try to find mkl.h anywhere
    try:
        result = subprocess.run(['find', '/usr', '-name', 'mkl.h', '-type', 'f'], 
                                capture_output=True, text=True, timeout=10)
        if result.stdout.strip():
            return os.path.dirname(result.stdout.strip().split('\n')[0])
    except:
        pass
    
    # Fallback - return empty, will fail at compile time if MKL is actually needed
    print("WARNING: MKL include path not found. Build may fail.")
    return ""

def get_compile_args():
    flags = ["-std=c++17", "-O1", "-fopenmp", "-Wno-ignored-qualifiers", "-mf16c"]
    info = cpuinfo.get_cpu_info()

    if 'avx512f' in info['flags']:
        flags.extend(["-mavx512f", "-mavx512cd", "-mavx512vl"])
    elif 'avx2' in info['flags']:
        flags.extend(["-mavx2", "-mfma"])
    return flags

ext_modules = []
ext_modules.append(
    Extension(
        name="fastmoe._cpu_kernel",
        sources=["fastmoe/csrc/flashattention.cpp"],
        include_dirs=torch.utils.cpp_extension.include_paths() + [get_mkl_include_path()],
        library_dirs=torch.utils.cpp_extension.library_paths() + [os.path.join(sys.prefix, 'lib')],
        libraries=['torch', 'c10', 'torch_cpu', 'torch_python', 'mkl_rt'],
        runtime_library_dirs=[os.path.join(sys.prefix, 'lib')],
        language="c++",
        extra_compile_args=get_compile_args(),
        extra_link_args=['-lpthread', '-lm', '-ldl']
    )
)

setup(
    name="FastMoE",
    version="0.0.1",
    packages=find_packages(
        exclude=("build", "include", "csrc", "test", "traces", "notebooks", "benchmarks", "fastmoe.egg-info")
    ),
    author="model toolchain",
    author_email="",
    description="Efficient MoE Inference",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
    python_requires=">=3.9",
    install_requires=[
        "aiohttp",
        "fastapi",
        "pyzmq",
        "rpyc",
        "torch>=2.1.2",
        "uvloop",
        "uvicorn",
        "psutil",
        "interegular",
        "lark",
        "numba",
        "pydantic",
        "referencing",
        "diskcache", 
        "cloudpickle",
        "pillow",
        "pulp",
        "numpy>=1.26.4",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
)