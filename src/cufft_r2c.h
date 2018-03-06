#ifndef CUFFT_R2C_H
#define CUFFT_R2C_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

  void DLL_EXPORT cufft_r2c(cufftHandle *plan,
                            const void *d_idata,
                            void *d_odata,
                            int dtype);
}


template<typename T>
inline cufftResult_t cufftTR2C(cufftHandle *plan,
                               const T *idata,
                               void *odata);


#endif /* ifndef CUFFT_R2C_H */
