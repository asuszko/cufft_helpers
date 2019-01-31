#ifndef CUFFT_C2C_H
#define CUFFT_C2C_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

  void DLL_EXPORT cufft_c2c(cufftHandle *plan,
                            void *idata,
                            void *odata,
                            int CUFFT_DIR,
                            int dtype);
}


#endif /* ifndef CUFFT_C2C_H */
