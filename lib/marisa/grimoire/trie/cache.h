#ifndef MARISA_GRIMOIRE_TRIE_CACHE_H_
#define MARISA_GRIMOIRE_TRIE_CACHE_H_

#include <cfloat>

#include "marisa/base.h"

namespace marisa {
namespace grimoire {
namespace trie {

class Cache {
 public:
  Cache() : parent_(0), child_(0), union_() {
    union_.weight = FLT_MIN;
  }
  Cache(const Cache &cache)
      : parent_(cache.parent_), child_(cache.child_), union_(cache.union_) {}

  Cache &operator=(const Cache &cache) {
    parent_ = cache.parent_;
    child_ = cache.child_;
    union_ = cache.union_;
    return *this;
  }

  void set_parent(std::size_t parent) {
    MARISA_DEBUG_IF(parent > MARISA_UINT32_MAX, MARISA_SIZE_ERROR);
    parent_ = (UInt32)parent;
  }
  void set_child(std::size_t child) {
    MARISA_DEBUG_IF(child > MARISA_UINT32_MAX, MARISA_SIZE_ERROR);
    child_ = (UInt32)child;
  }
  void set_base(Base base) {
    MARISA_DEBUG_IF(base > base_mask, MARISA_SIZE_ERROR);
    union_.link = (union_.link & ~base_mask) | base;
  }
  void set_extra(std::size_t extra, bool is_link) {
    MARISA_DEBUG_IF(extra > extra_limit, MARISA_SIZE_ERROR);
    union_.link = (UInt32)((union_.link & base_mask) | (extra << base_bits));
    if (is_link) {
       union_.link |= (link_mask + 1);
    }
  }
  void set_weight(float weight) {
    union_.weight = weight;
  }

  std::size_t parent() const {
    return parent_;
  }
  std::size_t child() const {
    return child_;
  }
  Base base() const {
    return (Base)(union_.link & base_mask);
  }
  std::size_t extra() const {
    return (union_.link & link_mask) >> base_bits;
  }
  Label label() const {
    return union_.link & link_mask;
  }
  std::size_t link() const {
    return union_.link & link_mask;
  }
  float weight() const {
    return union_.weight;
  }
  bool is_valid_link() const {
    return union_.link >> link_bits;
  }

 private:
  UInt32 parent_;
  UInt32 child_;
  union Union {
    UInt32 link;
    float weight;
  } union_;

  static constexpr UInt32 base_bits = sizeof(Base) * 8;
  static constexpr UInt32 base_mask = (1u << base_bits) - 1;
  static constexpr UInt32 link_bits = sizeof(UInt32) * 8 - 1;
  static constexpr UInt32 link_mask = (1u << link_bits) - 1;
  static constexpr UInt32 extra_limit = link_mask >> base_bits;
};

}  // namespace trie
}  // namespace grimoire
}  // namespace marisa

#endif  // MARISA_GRIMOIRE_TRIE_CACHE_H_
