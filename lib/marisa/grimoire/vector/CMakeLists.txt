set(LIB_VECTOR vector)
set(LIB_VECTOR_SRCS bit-vector.cc)

add_library(${LIB_VECTOR} STATIC ${LIB_VECTOR_SRCS})
target_include_directories(${LIB_VECTOR} PUBLIC /usr/local/cuda/include ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib)

set_source_files_properties(bit-vector.cc PROPERTIES LANGUAGE CUDA)
