#ifndef CUFFT_C2R_H
#define CUFFT_C2R_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

  void DLL_EXPORT cufft_c2r(cufftHandle *plan,
                            void *idata,
                            void *odata,
                            int dtype);
}


#endif /* ifndef CUFFT_C2R_H */
