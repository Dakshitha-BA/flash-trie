#ifndef MARISA_SCOPED_PTR_H_
#define MARISA_SCOPED_PTR_H_

#include <cuda_runtime_api.h>

#include "marisa/base.h"

namespace marisa {

template <typename T>
class scoped_ptr {
 public:
  scoped_ptr() : ptr_(NULL), managed_(false) {}
  explicit scoped_ptr(T *ptr, bool managed = false)
      : ptr_(ptr), managed_(managed) {}

  ~scoped_ptr() {
    if (managed_) {
      if (ptr_) {
        ptr_->~T();
      }
      cudaFree(ptr_);
    } else {
      delete ptr_;
    }
  }

  void reset(T *ptr = NULL, bool managed = false) {
    MARISA_DEBUG_IF((ptr != NULL) && (ptr == ptr_), MARISA_RESET_ERROR);
    scoped_ptr(ptr, managed).swap(*this);
  }

  __host__ __device__ T &operator*() const {
    MARISA_DEBUG_IF(ptr_ == NULL, MARISA_STATE_ERROR);
    return *ptr_;
  }
  __host__ __device__ T *operator->() const {
    MARISA_DEBUG_IF(ptr_ == NULL, MARISA_STATE_ERROR);
    return ptr_;
  }
  __host__ __device__ T *get() const {
    return ptr_;
  }

  void clear() {
    scoped_ptr().swap(*this);
  }
  void swap(scoped_ptr &rhs) {
    marisa::swap(ptr_, rhs.ptr_);
    marisa::swap(managed_, rhs.managed_);
  }

 private:
  T *ptr_;
  bool managed_;

  // Disallows copy and assignment.
  scoped_ptr(const scoped_ptr &);
  scoped_ptr &operator=(const scoped_ptr &);
};

}  // namespace marisa

#endif  // MARISA_SCOPED_PTR_H_
