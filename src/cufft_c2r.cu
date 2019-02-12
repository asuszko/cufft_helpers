#include <cuda.h>
#include <cufft.h>
#include "cu_errchk.h"
#include "cufft_c2r.h"



void cufft_c2r(cufftHandle *plan,
               void *idata,
               void *odata,
               int dtype)
{
    switch(dtype) {

        case 2:
            gpuFFTErrchk(cufftExecC2R(*plan,
                                      static_cast<float2*>(idata),
                                      static_cast<float*>(odata)));
            break;

        case 3:
            gpuFFTErrchk(cufftExecZ2D(*plan,
                                      static_cast<double2*>(idata),
                                      static_cast<double*>(odata)));
            break;
    }

    return;
}
