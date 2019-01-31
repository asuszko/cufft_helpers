#include <cuda.h>
#include <cufft.h>
#include "cu_errchk.h"
#include "cufft_setstream.h"

void cufft_setstream(cufftHandle *plan, cudaStream_t *stream)
{
    if (stream == NULL) {
        cufftSetStream(*plan, NULL);
    }
    else {
        cufftSetStream(*plan, *stream);
    }
    return;
}
