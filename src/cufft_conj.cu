#include <cuda.h>
#include <cufft.h>
#include "cu_errchk.h"
#include "cufft_conj.h"


template<typename T>
__global__ void complex_conj(T *odata,
	                           int nx,
                             int ny,
                             int nz)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int iz = threadIdx.z + blockDim.z * blockIdx.z;

    if (ix < nx && iy < ny && iz < nz) {
        odata[ix+iy*nx+iz*nx*ny].y *= -1.;
    }
}


void cufft_conj(void *d_data,
                dim3 extent,
                int dtype,
                cudaStream_t *stream)
{
    int nx = extent.x;
    int ny = extent.y;
    int nz = extent.z;

    dim3 blockSize;
    if (nz > 1) {
        (nz <= 16) ? blockSize.z = nz : blockSize.z = 16;
    }
    if (ny > 1) {
        (ny <= 16) ? blockSize.y = ny : blockSize.y = 16;
    }
    if (nx > 1) {
        (nx <= 16) ? blockSize.x = ny : blockSize.x = 16;
    }

    while(blockSize.z*blockSize.y*blockSize.x > 1024) {
        (blockSize.y > blockSize.z) ? blockSize.y /= 2 : blockSize.z /= 2;
    }

    dim3 gridSize((nx-1)/blockSize.x+1,
                  (ny-1)/blockSize.y+1,
                  (nz-1)/blockSize.z+1);

    switch(dtype) {

        case 2: {
            complex_conj<<<gridSize,blockSize,0,*stream>>>((float2*)d_data,nx,ny,nz);
            break;
        }

        case 3: {
            complex_conj<<<gridSize,blockSize,0,*stream>>>((double2*)d_data,nx,ny,nz);
            break;
        }
    }

    return;
}
