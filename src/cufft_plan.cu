#include <cuda.h>
#include <cufft.h>
#include "cu_errchk.h"
#include "cufft_plan.h"


#define CUFFTTYPE(type) ( (type) == 0 ? CUFFT_R2C : \
                          (type) == 1 ? CUFFT_C2R : \
                          (type) == 2 ? CUFFT_C2C : \
                          (type) == 3 ? CUFFT_D2Z : \
                          (type) == 4 ? CUFFT_Z2D : \
                          (type) == 5 ? CUFFT_Z2Z : CUFFT_Z2Z )


cufftHandle *cufft_plan(planlen extent,
                        int fft_type,
                        int batch_size)
{
	cufftHandle *plan = new cufftHandle;

    int ndims = 0;
    (extent[0] > 1) ? ndims += 1 : false;
    (extent[1] > 1) ? ndims += 1 : false;
    (extent[2] > 1) ? ndims += 1 : false;

    int *dims = new int[ndims];
    for (int i = 0; i < ndims; ++i) {
        dims[i] = extent[ndims-i-1];
    }

    gpuFFTErrchk(cufftPlanMany(plan, ndims, dims, NULL, 0, 0, NULL, 0, 0, CUFFTTYPE(fft_type), batch_size));
    cudaDeviceSynchronize();
    delete[] dims;

    return plan;
}


void cufft_plan_destroy(cufftHandle *plan)
{
    gpuFFTErrchk(cufftDestroy(*plan));
    delete[] plan;
}