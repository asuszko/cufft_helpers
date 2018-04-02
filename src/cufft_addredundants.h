#ifndef CUFFT_ADDREDUNDANTS_H
#define CUFFT_ADDREDUNDANTS_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

  void DLL_EXPORT cufft_addredundants(void *d_idata,
                                      void *d_odata,
                                      int nx, int ny,
                                      int dtype,
                                      cudaStream_t *stream=NULL);
}

#endif /* ifndef CUFFT_ADDREDUNDANTS_H */
