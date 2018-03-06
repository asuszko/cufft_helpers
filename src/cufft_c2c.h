#ifndef CUFFT_C2C_H
#define CUFFT_C2C_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

  void DLL_EXPORT cufft_c2c(cufftHandle *plan,
                            const void *d_idata,
                            void *d_odata,
                            int CUFFT_DIR,
                            int dtype);
}


template<typename T>
inline cufftResult_t cufftTC2C(cufftHandle *plan,
                               const T *idata,
                               void *odata,
                               int fft_dir);


#endif /* ifndef CUFFT_C2C_H */
