#ifndef CUFFT_PLAN_H
#define CUFFT_PLAN_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

typedef int planlen[3];

extern "C" {

  cufftHandle DLL_EXPORT *cufft_plan(planlen extent,
                                     int fft_type,
                                     int batch_size=1);
}


#endif /* ifndef CUFFT_PLAN_H */
