#include <cuda.h>
#include <cufft.h>
#include "cu_errchk.h"
#include "cufft_r2c.h"


template<typename T>
inline cufftResult_t cufftTR2C(cufftHandle *plan,
                               const T *idata,
                               void *odata)
{
    if (std::is_same<T, float>::value) {
        return cufftExecR2C(*plan,
                           (float *)idata,
                           (float2 *)odata);
    }
    else
    if (std::is_same<T, double>::value) {
        return cufftExecD2Z(*plan,
                           (double *)idata,
                           (double2 *)odata);
    }
    else {
        return CUFFT_EXEC_FAILED;
    }
}


/* C compatible version that requires a dtype_id to be converted
to the proper data type. */
void cufft_r2c(cufftHandle *plan,
               const void *d_idata,
               void *d_odata,
               int dtype)
{

    switch(dtype) {

        case 2: {
            gpuFFTErrchk(cufftTR2C(plan,
                                  (float*)d_idata,
                                  d_odata));
            break;
        }

        case 3: {
            gpuFFTErrchk(cufftTR2C(plan,
                                  (double*)d_idata,
                                  d_odata));
            break;
        }
    }

    return;
}
