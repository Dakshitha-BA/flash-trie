cmake_minimum_required(VERSION 3.14)
project(marisa_tools LANGUAGES CXX)

find_package(pybind11 REQUIRED)

set(PROGS
  marisa-build
  marisa-lookup
  marisa-reverse-lookup
  marisa-common-prefix-search
  marisa-predictive-search
  marisa-dump
  marisa-benchmark
  marisa-beam-search
)

foreach(prog ${PROGS})
  add_executable(${prog} ${prog}.cc cmdopt.cc)
  target_include_directories(${prog} PUBLIC /usr/local/cuda/include)
  target_link_libraries(${prog} PRIVATE marisa)
endforeach()

set(BIND_SRCS bindings.cpp)

pybind11_add_module(flashtrie ${BIND_SRCS})

target_compile_options(flashtrie PUBLIC -O3 -std=c++17 -Wall -Wextra -fmax-errors=1)

target_include_directories(flashtrie PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  /usr/local/cuda/include
)

target_link_libraries(flashtrie PRIVATE marisa)
