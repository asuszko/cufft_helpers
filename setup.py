# -*- coding: utf-8 -*-

import os
import os.path as path
import platform
import subprocess


def build_cuda(wd, name, sources,
               compile_args=["-arch=sm_50", "-m64", "-Xcompiler", "-fPIC", "-std=c++11"],
               include_dirs=[],
               library_dirs=[],
               libraries=[],
               extra_compile_args=[]):

    """
    Compile sources to shared library using nvcc.
    """
    
    if platform.system() == "Linux":
        ext = ".so"
        cc_bin = None
    elif platform.system() == "Windows":
        ext = ".dll"
        cc_bin = r'"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"'
    
    cwd = os.getcwd()
    os.chdir(wd)

    as_list = lambda str_or_list: [str_or_list] if isinstance(str_or_list, str) else str_or_list

    sources = as_list(sources)
    compile_args = as_list(compile_args)
    include_dirs = as_list(include_dirs)
    library_dirs = as_list(library_dirs)
    libraries = as_list(libraries)
    extra_compile_args = as_list(extra_compile_args)

    call_list = ["nvcc"]
    call_list += compile_args
    call_list += ["-shared", "-o", name+ext]
    call_list += sources
    call_list += ["-I"+I for I in include_dirs]
    call_list += ["-L"+L for L in library_dirs]
    if cc_bin is not None:
        call_list += [as_list("-ccbin "+cc_bin)]
    call_list += ["-l"+l for l in libraries]
    call_list += extra_compile_args

    print(subprocess.list2cmdline(call_list))
    subprocess.call(call_list)

    os.chdir(cwd)




# Relevant paths
base_path = path.dirname(path.realpath(__file__))
src_path = path.join(base_path,"src")
lib_path = path.join(base_path,"lib")

# Create the folder to put the shared library in
if not os.path.exists(lib_path):
    os.makedirs(lib_path)

# Compile into shared library
build_cuda(src_path, path.join(lib_path,"cufft"),[
           "cufft_addredundants.cu",
           "cufft_r2c.cu",
           "cufft_c2c.cu",
           "cufft_c2r.cu",
           "cufft_conj.cu",
           "cufft_plan.cu",
           "cufft_setstream.cu",
           ],
           libraries=["cuda", "cufft"])