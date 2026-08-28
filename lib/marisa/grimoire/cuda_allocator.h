#pragma once

#include <cuda_runtime_api.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/atomic>
#include <iostream>
#include <cassert>
#include <cuda.h>

#include "cuda_check.h"

namespace marisa {
namespace grimoire {

template <class T>
struct managed_allocator {
  typedef cuda::std::size_t size_type;
  typedef cuda::std::ptrdiff_t difference_type;

  typedef T value_type;
  typedef T* pointer;// (deprecated in C++17)(removed in C++20) T*
  typedef const T* const_pointer;// (deprecated in C++17)(removed in C++20) const T*
  typedef T& reference;// (deprecated in C++17)(removed in C++20) T&
  typedef const T& const_reference;// (deprecated in C++17)(removed in C++20) const T&

  template< class U > struct rebind { typedef managed_allocator<U> other; };
  managed_allocator() = default;
  template <class U> constexpr managed_allocator(const managed_allocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    void* out = nullptr;
    CUDA_CHECK(cudaMallocManaged(&out, n*sizeof(T)));
    return static_cast<T*>(out);
  }
  void deallocate(T* p, std::size_t) noexcept {
    CUDA_CHECK(cudaFree(p));
  }
};

template <typename T, typename U>
bool operator==(const managed_allocator<T>&, const managed_allocator<U>&) {
  return true;
}
template <typename T, typename U>
bool operator!=(const managed_allocator<T>&, const managed_allocator<U>&) {
  return false;
}

template<class T, class... Args>
T* make_managed(Args &&... args) {
    managed_allocator<T> ma;
    return new (ma.allocate(1)) T(std::forward<Args>(args)...);
}

template <class T>
struct pinned_allocator {
  typedef cuda::std::size_t size_type;
  typedef cuda::std::ptrdiff_t difference_type;

  typedef T value_type;
  typedef T* pointer;// (deprecated in C++17)(removed in C++20) T*
  typedef const T* const_pointer;// (deprecated in C++17)(removed in C++20) const T*
  typedef T& reference;// (deprecated in C++17)(removed in C++20) T&
  typedef const T& const_reference;// (deprecated in C++17)(removed in C++20) const T&

  template< class U > struct rebind { typedef pinned_allocator<U> other; };
  pinned_allocator() = default;
  template <class U> constexpr pinned_allocator(const pinned_allocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    void* out = nullptr;
    CUDA_CHECK(cudaMallocHost(&out, n*sizeof(T)));
    return static_cast<T*>(out);
  }
  void deallocate(T* p, std::size_t) noexcept {
    CUDA_CHECK(cudaFreeHost(p));
  }
};

template<class T>
using managed_vector = std::vector<T, managed_allocator<T>>;
template<class T>
using pinned_vector = std::vector<T, pinned_allocator<T>>;

}  // namespace grimoire
}  // namespace marisa
