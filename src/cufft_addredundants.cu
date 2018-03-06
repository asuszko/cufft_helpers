#include <cuda.h>
#include <cufft.h>
#include "cu_errchk.h"
#include "cufft_addredundants.h"

/* This will generate an array that adds the redundant values of a
CUDA R2C transformation. This function was found here:
https://devtalk.nvidia.com/default/topic/488433/cufft-only-gives-non-redundant-results/
*/
template<typename T>
__global__ void k_makeRedundant(T* dst, const T* src, int w, int h)
{
    volatile int gid_x = threadIdx.x + blockIdx.x * blockDim.x;
    volatile int gid_y = threadIdx.y + blockIdx.y * blockDim.y;
    volatile int nbNoRedundants = (w >> 1) + 1;

    // index for reading :
    volatile int gid = gid_x + nbNoRedundants * gid_y;
    T val;

    if(gid_x < nbNoRedundants && gid_y < h) {
        // write the non redundant part in the new array :
        val = src[gid];
        gid = gid_x + w * gid_y; // new index for writing
        dst[gid] = val;
    }

    // shift'n'flip
    gid_x = w - gid_x;

    if(gid_y != 0) {
        gid_y = h - gid_y;
    }

    gid = gid_x + w * gid_y;

    // write conjugate :

    if(gid_x >= nbNoRedundants && gid_x < w && gid_y >= 0 && gid_y < h) {
        val.y = -val.y;
        dst[gid] = val; // never coalesced with compute <= 1.1 ; coalesced if >= 1.2 AND w multiple of 16 AND good call configuration
    }
}


/* C compatible version that requires a dtype_id to be converted
to the proper data type. */
void cufft_addredundants(const void *d_idata,
                         void *d_odata,
                         int nx, int ny,
                         int dtype,
                         cudaStream_t *stream)
{
    dim3 blockSize(16,16);
    dim3 gridSize((nx-1)/blockSize.x+1,
                  (ny-1)/blockSize.y+1);

    cudaStream_t stream_id;
    (stream == NULL) ? stream_id = NULL : stream_id = *stream;

    switch(dtype) {

        case 2: {
            k_makeRedundant<<<gridSize, blockSize, 0, stream_id>>>((float2*)d_odata,
                                                                   (float2*)d_idata,
                                                                   nx, ny);
            break;
        }

        case 3: {
            k_makeRedundant<<<gridSize, blockSize, 0, stream_id>>>((double2*)d_odata,
                                                                   (double2*)d_idata,
                                                                   nx, ny);
            break;
        }
    }

    return;
}
