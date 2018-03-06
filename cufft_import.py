# -*- coding: utf-8 -*-

__all__ = [
    "cufft_conj",
    "cufft_c2c",
    "cufft_c2r",
    "cufft_plan",
    "cufft_r2c",
    "cufft_setstream",
]

import os
import sys
from numpy.ctypeslib import load_library, ndpointer
from ctypes import (c_bool,
                    c_int,
                    c_void_p,
                    c_size_t)

# Load the shared library
sys.path.append("..")
from shared_utils.load_lib import load_lib
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib"))
cufft_lib = load_lib(lib_path,"cufft")


# Define argtypes for all functions to import
argtype_defs = {

    "cufft_addredundants" : [c_void_p,      #Device pointer to non-redundant data
                             c_void_p,      #Device pointer to redundant data
                             c_int, c_int,  #Dimensions of the array
                             c_int,         #Data type identifier
                             c_void_p],     #Pointer to CUDA stream


    "cufft_c2c" : [c_void_p,                #Pointer to the cuFFT plan
                   c_void_p,                #Device pointer to the input data
                   c_void_p,                #Device pointer to the output data
                   c_int,                   #Direction of the fft (forward or inverse)
                   c_int],                  #Data type identifier


    "cufft_c2r" : [c_void_p,                #Pointer to the cuFFT plan
                   c_void_p,                #Device pointer to the input data
                   c_void_p,                #Device pointer to the output data
                   c_int],                  #Data type identifier               
 
                    
    "cufft_r2c" : [c_void_p,                #Pointer to the cuFFT plan
                   c_void_p,                #Device pointer to the input data
                   c_void_p,                #Device pointer to non-redundant data
                   c_int],                  #Data type identifier

    "cufft_conj" : [c_void_p,               #Device pointer to the input data
                    ndpointer(),            #Dimensions of the input data
                    c_int,                  #Data type identifier
                    c_void_p],              #Pointer to CUDA stream
                    
                    
    "cufft_plan" : [ndpointer(),            #Dimensions of the fft
                    c_int,                  #Type of fft
                    c_int],                 #Batch size of fft
                    
    "cufft_setstream" : [c_void_p,          #Pointer to the cuFFT plan
                         c_void_p]          #Pointer to CUDA stream
}

restype_defs = {

    "cufft_plan": c_void_p

}


# Import functions from DLL
for func, argtypes in argtype_defs.items():
    restype = restype_defs.get(func)
    vars().update({func: cufft_lib[func]})
    vars()[func].argtypes = argtypes
    vars()[func].restype = restype