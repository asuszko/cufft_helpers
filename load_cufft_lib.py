# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:11:00 2018

@author: pj8pgp
"""

import os
import platform
from numpy.ctypeslib import load_library

def load_cufft_lib(lib_rel_dir, lib_fname):
    wdir = os.path.dirname( __file__ )
    lib_path = os.path.abspath(os.path.join(wdir, lib_rel_dir))
    if platform.system() == 'Linux':
        c_lib = load_library(lib_fname+".so", lib_path)
    elif platform.system() == 'Windows':
        c_lib = load_library(lib_fname+".dll", lib_path)
    return c_lib