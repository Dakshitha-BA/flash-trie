cmake_minimum_required(VERSION 3.5)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(marisa-trie LANGUAGES CXX CUDA)

file(READ "${CMAKE_SOURCE_DIR}/VERSION" FLASHTRIE_VERSION)
string(STRIP "${FLASHTRIE_VERSION}" FLASHTRIE_VERSION)
message(STATUS "FlashTrie version: ${FLASHTRIE_VERSION}")

# Prefer the toolkit CMake found. The hardcoded /usr/local/cuda include in
# subdirectories remains as a fallback for the existing developer build.
if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# Optional: record a per-phase GPU timing breakdown (beam expansion, child
# validation, top-B selection, grid-sync/launch overhead) in the beam search
# kernel. Enable with: cmake -DTBS_PROFILE=ON ..
option(TBS_PROFILE "Record per-phase GPU timing breakdown in beam search" OFF)
if(TBS_PROFILE)
  add_compile_definitions(TBS_PROFILE)
  message(STATUS "TBS_PROFILE enabled: beam search will record per-phase timing")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native -std=c++17 -fmax-errors=1 -Wall -Wconversion -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMARISA_USE_POPCNT -mpopcnt -DMARISA_USE_BMI2 -mbmi2 -msse4")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-g,-O3,-march=native,-std=c++17,-fmax-errors=1,-Wall,-Wconversion,-fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-DMARISA_USE_POPCNT,-mpopcnt,-DMARISA_USE_BMI2,-mbmi2,-msse4")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr")

add_subdirectory(lib/marisa)
add_subdirectory(tools)
