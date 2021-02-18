set(ATen_HIP_INCLUDE ${ATen_HIP_INCLUDE}
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
PARENT_SCOPE)

configure_file(THHGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THHGeneral.h")

set(extra_src)
# loop over all types
foreach(THH_TYPE Byte Char Short Int Long Half Float Double)
   # loop over files which need to be split between types (because of long compile times)
   foreach(THH_FILE TensorSort TensorMathPointwise TensorMathReduce TensorMasked TensorTopK)
      if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/THH${THH_FILE}${THH_TYPE}.hip")
         file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/generated/THH${THH_FILE}${THH_TYPE}.hip"
           "#include <THH/THH${THH_FILE}.cuh>\n#include <THH/THHTensor.hpp>\n\n#include <THH/generic/THH${THH_FILE}.hip>\n#include <THH/THHGenerate${THH_TYPE}Type.h>\n")
      endif()
      list(APPEND extra_src "${CMAKE_CURRENT_SOURCE_DIR}/generated/THH${THH_FILE}${THH_TYPE}.hip")
   endforeach()
endforeach()

foreach(THH_FILE TensorMathPointwise TensorMathReduce TensorMasked)
   if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/THH${THH_FILE}Bool.hip")
      file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/generated/THH${THH_FILE}Bool.hip"
        "#include <THH/THH${THH_FILE}.cuh>\n#include <THH/THHTensor.hpp>\n\n#include <THH/generic/THH${THH_FILE}.hip>\n#include <THH/THHGenerateBoolType.h>\n")
   endif()
   list(APPEND extra_src "${CMAKE_CURRENT_SOURCE_DIR}/generated/THH${THH_FILE}Bool.hip")
endforeach()

foreach(THH_FILE TensorMathReduce TensorMasked TensorSort TensorTopK)
   if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/THH${THH_FILE}BFloat16.hip")
      file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/generated/THH${THH_FILE}BFloat16.hip"
        "#include <THH/THH${THH_FILE}.cuh>\n#include <THH/THHTensor.hpp>\n\n#include <THH/generic/THH${THH_FILE}.hip>\n#include <THH/THHGenerateBFloat16Type.h>\n")
   endif()
   list(APPEND extra_src "${CMAKE_CURRENT_SOURCE_DIR}/generated/THH${THH_FILE}BFloat16.hip")
endforeach()

set(ATen_HIP_SRCS ${ATen_HIP_SRCS}
  ${CMAKE_CURRENT_SOURCE_DIR}/THHCachingHostAllocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THHGeneral.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THHStorageCopy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensor.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/THHReduceApplyUtils.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHSleep.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHStorage.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHStorageCopy.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensor.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorCopy.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorMath.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorMathMagma.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorMathPairwise.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorMathReduce.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorIndex.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorRandom.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorScatterGather.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorSort.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHSortUtils.hip
  ${extra_src}
  PARENT_SCOPE)

set(ATen_HIP_SRCS_W_SORT_BY_KEY ${ATen_HIP_SRCS_W_SORT_BY_KEY}
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensorMode.hip
  PARENT_SCOPE)

install(FILES
          THH.h
          ${CMAKE_CURRENT_BINARY_DIR}/THHGeneral.h
          THHGeneral.hpp
          THHSleep.h
          THHStorage.h
          THHStorageCopy.h
          THHTensor.h
          THHTensorCopy.h
          THHTensorCopy.hpp
          THHTensorRandom.h
          THHTensorMath.h
          THHApply.cuh
          THHReduce.cuh
          THHReduceAll.cuh
          THHReduceApplyUtils.cuh
          THHTensorMathReduce.cuh
          THHAsmUtils.cuh
          THHAtomics.cuh
          THHScanUtils.cuh
          THHSortUtils.cuh
          THHAllocator.h
          THHCachingHostAllocator.h
          THHDeviceUtils.cuh
          THHDeviceTensor.cuh
          THHDeviceTensor-inl.cuh
          THHDeviceTensorUtils.cuh
          THHDeviceTensorUtils-inl.cuh
          THHGenerateAllTypes.h
          THHGenerateBFloat16Type.h
          THHGenerateBoolType.h
          THHGenerateByteType.h
          THHGenerateCharType.h
          THHGenerateShortType.h
          THHGenerateIntType.h
          THHGenerateLongType.h
          THHGenerateHalfType.h
          THHGenerateFloatType.h
          THHGenerateFloatTypes.h
          THHGenerateDoubleType.h
          THHGenerateComplexFloatType.h
          THHGenerateComplexTypes.h
          THHGenerateComplexDoubleType.h
          THHIntegerDivider.cuh
          THHNumerics.cuh
          THHTensorSort.cuh
          THHTensorInfo.cuh
          THHTensorMathPointwise.cuh
          THHTensorTypeUtils.cuh
          THHTensorRandom.cuh
          THHTensorMathMagma.cuh
          THHThrustAllocator.cuh
          THHTensorMode.cuh
          THHTensorTopK.cuh
          # See Note [TH abstraction violation]
          THHTensor.hpp
          THHStorage.hpp
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THH")

install(FILES
          generic/THHStorage.cpp
          generic/THHStorage.hip
          generic/THHStorage.h
          generic/THHTensor.cpp
          generic/THHTensor.hip
          generic/THHTensor.h
          generic/THHTensor.hpp
          generic/THHStorageCopy.cpp
          generic/THHStorageCopy.hip
          generic/THHStorageCopy.h
          generic/THHTensorCopy.hip
          generic/THHTensorCopy.h
          generic/THHTensorMasked.h
          generic/THHTensorMasked.hip
          generic/THHTensorMath.h
          generic/THHTensorMath.hip
          generic/THHTensorMathMagma.h
          generic/THHTensorMathMagma.hip
          generic/THHTensorMathPairwise.h
          generic/THHTensorMathPairwise.hip
          generic/THHTensorMathPointwise.h
          generic/THHTensorMathPointwise.hip
          generic/THHTensorMathReduce.h
          generic/THHTensorMathReduce.hip
          generic/THHTensorScatterGather.h
          generic/THHTensorScatterGather.hip
          generic/THHTensorIndex.h
          generic/THHTensorIndex.hip
          generic/THHTensorSort.h
          generic/THHTensorSort.hip
          generic/THHTensorRandom.h
          generic/THHTensorRandom.hip
          generic/THHTensorMode.h
          generic/THHTensorMode.hip
          generic/THHTensorTopK.h
          generic/THHTensorTopK.hip
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THH/generic")
