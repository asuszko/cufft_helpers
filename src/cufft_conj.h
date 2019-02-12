#ifndef CUFFT_CONJ_H
#define CUFFT_CONJ_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

  void DLL_EXPORT cufft_conj(void *d_data,
                             dim3 extent,
                             int dtype,
                             cudaStream_t *stream=NULL);
}


#endif /* ifndef CUFFT_CONJ_H */
