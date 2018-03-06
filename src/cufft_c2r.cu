#include <cuda.h>
#include <cufft.h>
#include "cu_errchk.h"
#include "cufft_c2r.h"


template<typename T>
inline cufftResult_t cufftTC2R(cufftHandle *plan,
                               const T *idata,
                               void *odata)
{
    if (std::is_same<T, float2>::value) {
        return cufftExecC2R(*plan,
                           (float2 *)idata,
                           (float *)odata);
    }
    else
    if (std::is_same<T, double2>::value) {
        return cufftExecZ2D(*plan,
                           (double2 *)idata,
                           (double *)odata);
    }
    else {
        return CUFFT_EXEC_FAILED;
    }
}


/* C compatible version that requires a dtype_id to be converted
to the proper data type. */
void cufft_c2r(cufftHandle *plan,
               const void *d_idata,
               void *d_odata,
               int dtype)
{
    switch(dtype) {

        case 2: {
            gpuFFTErrchk(cufftTC2R(plan,
                                  (float2 *)d_idata,
                                  d_odata));
            break;
        }

        case 3: {
            gpuFFTErrchk(cufftTC2R(plan,
                                  (double2 *)d_idata,
                                  d_odata));
            break;
        }
    }

    return;
}
