# -*- coding: utf-8 -*-

import glob
import os
import platform
import subprocess
import sys

__lib_name = "cufft"


def build_cuda(wd, name, sources,
               arch="-arch=sm_30",
               compile_args=["-m64", "-Xcompiler", "-fPIC", "-std=c++11"],
               cl_bin=None,
               include_dirs=[],
               library_dirs=[],
               libraries=[],
               extra_compile_args=[]):

    """
    Compile sources to shared library using nvcc.
    """
    
    if platform.system() == "Linux":
        ext = ".so"
        cl_bin=[]
    elif platform.system() == "Windows":
        ext = ".dll"
    
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
    call_list += [arch]
    call_list += compile_args
    call_list += ["-shared", "-o", name+ext]
    call_list += sources
    call_list += ["-I"+I for I in include_dirs]
    call_list += ["-L"+L for L in library_dirs]
    call_list += [as_list("-ccbin "+cl_bin)]
    call_list += ["-l"+l for l in libraries]
    call_list += extra_compile_args

    print(subprocess.list2cmdline(call_list))
    subprocess.call(call_list)

    os.chdir(cwd)




def main():
    
    for arg in sys.argv[1:]:
        if "-cc_bin" in arg:
            cc_bin = '"'+arg[8:]+'"'
        if "-arch" in arg:
            arch = arg
    
    base_path = os.path.abspath(os.path.dirname(__file__))
    src_path = os.path.join(base_path,"src")
    lib_path = os.path.join(base_path,"lib")
    
    # Create the folder if not exists to put shared library in
    if not os.path.exists(lib_path):
        os.makedirs(lib_path)
    
    # Compile into shared library
    os.chdir(src_path)
    build_cuda(src_path,
               os.path.join(lib_path,__lib_name),
               glob.glob("*.cu"),
               arch=arch,
               cl_bin=cc_bin,
               libraries=["cuda", "cufft"])
    
     #Cleanup extra compile files
    os.chdir(lib_path)
    os.remove(__lib_name+".exp")
    os.remove(__lib_name+".lib")
    
    
if __name__ == "__main__":
    main()