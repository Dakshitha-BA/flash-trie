set(LIB_IO io)
set(LIB_IO_SRCS mapper.cc reader.cc writer.cc)

add_library(${LIB_IO} STATIC ${LIB_IO_SRCS})
target_include_directories(${LIB_IO} PUBLIC /usr/local/cuda/include ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib)
