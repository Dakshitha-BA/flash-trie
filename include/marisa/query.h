#ifndef MARISA_QUERY_H_
#define MARISA_QUERY_H_

#include "marisa/base.h"

namespace marisa {

class Query {
 public:
  Query() : ptr_(NULL), length_(0), id_(0) {}
  Query(const Query &query)
      : ptr_(query.ptr_), length_(query.length_), id_(query.id_) {}

  Query &operator=(const Query &query) {
    ptr_ = query.ptr_;
    length_ = query.length_;
    id_ = query.id_;
    return *this;
  }

  __host__ __device__ Label operator[](std::size_t i) const {
    MARISA_DEBUG_IF(i >= length_, MARISA_BOUND_ERROR);
    return ptr_[i];
  }

  void set_str(const std::vector<Label>& str) {
    set_str(str.data(), str.size());
  }
  __host__ __device__ void set_str(const Label *ptr, std::size_t length) {
    MARISA_DEBUG_IF((ptr == NULL) && (length != 0), MARISA_NULL_ERROR);
    ptr_ = ptr;
    length_ = length;
  }

  __host__ __device__ void set_id(std::size_t id) {
    id_ = id;
  }

  __host__ __device__ const Label *ptr() const {
    return ptr_;
  }

  __host__ __device__ std::size_t length() const {
    return length_;
  }
  __host__ __device__ std::size_t id() const {
    return id_;
  }
  std::vector<Label> vec() const {
    return std::vector<Label>(ptr_, ptr_ + length_);
  }

  void clear() {
    Query().swap(*this);
  }
  void swap(Query &rhs) {
    marisa::swap(ptr_, rhs.ptr_);
    marisa::swap(length_, rhs.length_);
    marisa::swap(id_, rhs.id_);
  }

 private:
  const Label *ptr_;
  std::size_t length_;
  std::size_t id_;
};

}  // namespace marisa

#endif  // MARISA_QUERY_H_
