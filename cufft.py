# -*- coding: utf-8 -*-
__all__ = [
    "cufft",
]

import numpy as np

# Local imports
from cufft_import import (cufft_addredundants,
                          cufft_c2c,
                          cufft_c2r,
                          cufft_conj,
                          cufft_plan,
                          cufft_r2c,
                          cufft_setstream)

_type2id_types = {np.dtype('f4'):0,
                  np.dtype('f8'):1,
                  np.dtype('c8'):2,
                  np.dtype('c16'):3}

_cufft_type_map = {
    "CUFFT_R2C": 0,  # single precision, real -> complex
    "CUFFT_C2R": 1,  # single precision, complex -> real
    "CUFFT_C2C": 2,  # single precision, complex -> complex
    "CUFFT_D2Z": 3,  # double precision, real -> complex
    "CUFFT_Z2D": 4,  # double precision, complex -> real
    "CUFFT_Z2Z": 5,  # double precision, complex -> complex
}

# Backend C-code uses a switch statement with either 2 or 3
_cufft_dtype_map = {
    "CUFFT_R2C": 2,
    "CUFFT_C2R": 2,
    "CUFFT_C2C": 2,
    "CUFFT_D2Z": 3,
    "CUFFT_Z2D": 3,
    "CUFFT_Z2Z": 3,
}

_cufft_dir_map = {
    "CUFFT_FORWARD": 0,
    "CUFFT_INVERSE": 1,
}



class cufft(object):

    def __init__(self, stream=None):
        self._stream = stream


    def plan(self, extent, fft_type, batch_size=1):
        """
        Returns a dict containing what's needed to 
        perform the proper cuFFT.
        
        Parameters
        ----------
        extent : 1D array
            dimensions of fft: nx, ny, nz
    
        fft_type : str or int
            Type of cufft defined in _cufft_type_map
    
        batch_size : int
            Number of ffts to perform in batch mode
            
        Returns
        -------
        dict : 
            A dict with the plan pointer, type of fft, 
            and fft dimensions.
        """
        
        # Get the fft_type
        cufft_type = _cufft_type_map[fft_type.upper()]
            
        # Create the plan and return the pointer to the plan            
        plan_ptr = cufft_plan(extent, cufft_type, batch_size)

        if self.stream is not None:
            cufft_setstream(plan_ptr, self.stream)
            
        return {'ptr': plan_ptr,
                'plan_id': _cufft_dtype_map[fft_type.upper()],
                'extent': extent}


    def add_redundants(self, plan, idata, odata):
        """
        This is not an official cuFFT function, but may be 
        useful in the cuFFT context. cuFFT r2c calls do not 
        include redundants in the result. This function 
        brings them back.
        
        Parameters
        ----------
        plan : dict
            A dict containing the plan information.
            
        idata : c_void_p
            Device pointer to the non-redundant input data.
            
        odata: c_void_p
            Device pointer to the output data with redundants.
        """
        cufft_addredundants(idata, odata,
                            plan['extent'][0], 
                            plan['extent'][1],
                            plan['plan_id'],
                            self.stream)
        

    def c2c(self, plan, idata, odata, fft_dir):
        """
        Executes a complex-to-complex transform plan in the 
        transform direction as specified by direction parameter. 
        cuFFT uses the GPU memory pointed to by the idata parameter 
        as input data. This function stores the Fourier coefficients 
        in the odata array. If idata and odata are the same, 
        this method does an in-place transform.
        
        Parameters
        ----------
        plan : dict
            A dict containing the plan information.
            
        idata : c_void_p
            Device pointer to the complex input data.
            
        odata : c_void_p
            Device pointer to the complex output data.
            
        fft_dir : int or str
            The transform direction: 'CUFFT_FORWARD' or 'CUFFT_INVERSE'.
        """
        if fft_dir is not int:
            cufft_dir = _cufft_dir_map[fft_dir.upper()]
        else:
            cufft_dir = fft_dir
        cufft_c2c(plan['ptr'], idata, odata, cufft_dir, plan['plan_id'])


    def c2r(self, plan, idata, odata):
        """
        Executes a complex-to-real, implicitly inverse, cuFFT transform 
        plan. cuFFT uses as input data the GPU memory pointed to by 
        the idata parameter. The input array holds only the 
        nonredundant complex Fourier coefficients. This function 
        stores the real output values in the odata array. If idata and 
        odata are the same, this method does an in-place transform.
        
        Parameters
        ----------
        plan : dict
            A dict containing the plan information.
            
        idata : c_void_p
            Device pointer to the complex input data.
            
        odata : c_void_p
            Device pointer to the real output data.
        """
        cufft_c2r(plan['ptr'], idata, odata, plan['plan_id'])


    def r2c(self, plan, idata, odata):
        """
        Executes a real-to-complex, implicitly forward, cuFFT transform 
        plan. cuFFT uses as input data the GPU memory pointed to by 
        the idata parameter. This function stores the nonredundant 
        Fourier coefficients in the odata array. If idata and odata are 
        the same, this method does an in-place transform.
        
        Parameters
        ----------
        plan : dict
            A dict containing the plan information.
            
        idata : c_void_p
            Device pointer to the real input data.
            
        odata : c_void_p
            Device pointer to the complex output data.
        """
        cufft_r2c(plan['ptr'], idata, odata, plan['plan_id'])


    def complex_conj(self, data, extent, dtype_id):
        """
        Turns a device array into its complex conjugate.
        
        Parameters
        ----------
        data : c_void_p
            Device pointer to complex data.
            
        extent : list or np.ndarray (1d)
            Dimensions of the data array.
            
        dtype_id : int or np.dtype
            Data type identifier
        """
        if dtype_id is np.ndarray:
            dtype_id = self.types[dtype_id]
        cufft_conj(data, extent, dtype_id, self.stream)
        

    @property
    def stream(self):
        return self._stream
        
        
    @property
    def types(self):
        return _type2id_types