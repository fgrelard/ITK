project(ITKFFT)

# this library is only needed if FFTW is used
if(ITK_USE_FFTWF OR ITK_USE_FFTWD)
  if(ITK_USE_CUFFTW)
    set(ITKFFT_LIBRARIES ${FFTWF_LIB})
  else()
    set(ITKFFT_LIBRARIES ITKFFT)
  endif()
  set(ITKFFT_INCLUDE_DIRS ${FFTW_INCLUDE_PATH})
endif()
itk_module_impl()
