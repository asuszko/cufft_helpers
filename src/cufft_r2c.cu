#include <cuda.h>
#include <cufft.h>
#include "cu_errchk.h"
#include "cufft_r2c.h"


void cufft_r2c(cufftHandle *plan,
               void *d_idata,
               void *d_odata,
               int dtype)
{
    switch(dtype) {

        case 2:
            gpuFFTErrchk(cufftExecR2C(*plan,
                                      static_cast<float*>(d_idata),
                                      static_cast<float2*>(d_odata)));
            break;

        case 3:
            gpuFFTErrchk(cufftExecD2Z(*plan,
                                      static_cast<double*>(d_idata),
                                      static_cast<double2*>(d_odata)));
            break;
    }

    return;
}
