#include <cuda.h>
#include <cufft.h>
#include "cu_errchk.h"
#include "cufft_c2c.h"


#define CUFFTDIR(dir) ( (dir) == 0 ? CUFFT_FORWARD : \
                        (dir) == 1 ? CUFFT_INVERSE : CUFFT_INVERSE)


template<typename T>
inline cufftResult_t cufftTC2C(cufftHandle *plan,
                               const T *idata,
                               void *odata,
                               int fft_dir)
{
    if (std::is_same<T, float2>::value) {
        return cufftExecC2C(*plan,
                           (float2 *)idata,
                           (float2 *)odata,
                           CUFFTDIR(fft_dir));
    }
    else
    if (std::is_same<T, double2>::value) {
        return cufftExecZ2Z(*plan,
                           (double2 *)idata,
                           (double2 *)odata,
                           CUFFTDIR(fft_dir));
    }
    else {
        return CUFFT_EXEC_FAILED;
    }
}



/* C compatible version that requires a dtype_id to be converted
to the proper data type. */
void cufft_c2c(cufftHandle *plan,
               const void *d_idata,
               void *d_odata,
               int CUFFT_DIR,
               int dtype)
{
    switch(dtype) {

          case 2: {
              gpuFFTErrchk(cufftTC2C(plan,
                                    (float2*)d_idata,
                                    d_odata,
                                    CUFFT_DIR));
              break;
          }

          case 3: {
              gpuFFTErrchk(cufftTC2C(plan,
                                    (double2*)d_idata,
                                    d_odata,
                                    CUFFT_DIR));
              break;
          }
      }

      return;
}
