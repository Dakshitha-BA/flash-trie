#pragma once
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdio>

#define CUDA_CHECK(ans) \
  { cuda_check((ans), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
  if (code == cudaSuccess) return;
  printf("cuda operation failed with error %s in %s at %d\n",
      cudaGetErrorString(code), file, line);
  assert(false);
}
