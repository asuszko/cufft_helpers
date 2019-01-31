#include <cuda.h>
#include <cufft.h>
#include "cu_errchk.h"
#include "cufft_c2c.h"


#define CUFFTDIR(dir) ( (dir) == 0 ? CUFFT_FORWARD : \
                        (dir) == 1 ? CUFFT_INVERSE : CUFFT_INVERSE)



void cufft_c2c(cufftHandle *plan,
               void *idata,
               void *odata,
               int CUFFT_DIR,
               int dtype)
{
    switch(dtype) {

        case 2:
            gpuFFTErrchk(cufftExecC2C(*plan,
                                      static_cast<float2*>(idata),
                                      static_cast<float2*>(odata),
                                      CUFFTDIR(CUFFT_DIR)));
            break;

        case 3:
            gpuFFTErrchk(cufftExecZ2Z(*plan,
                                      static_cast<double2*>(idata),
                                      static_cast<double2*>(odata),
                                      CUFFTDIR(CUFFT_DIR)));
            break;
    }

    return;
}
