#ifndef CUFFT_C2R_H
#define CUFFT_C2R_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

  void DLL_EXPORT cufft_c2r(cufftHandle *plan,
                            const void *d_idata,
                            void *d_odata,
                            int dtype);
}


template<typename T>
inline cufftResult_t cufftTC2R(cufftHandle *plan,
                               const T *idata,
                               void *odata);


#endif /* ifndef CUFFT_C2R_H */
