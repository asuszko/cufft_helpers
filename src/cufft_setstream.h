#ifndef CUFFT_SETSTREAM_H
#define CUFFT_SETSTREAM_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

  void DLL_EXPORT cufft_setstream(cufftHandle *plan,
                                  cudaStream_t *stream);
}


#endif /* ifndef CUFFT_SETSTREAM_H */
