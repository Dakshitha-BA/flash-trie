#ifndef MARISA_GRIMOIRE_TRIE_TAIL_H_
#define MARISA_GRIMOIRE_TRIE_TAIL_H_

#include "marisa/agent.h"
#include "marisa/grimoire/vector.h"
#include "marisa/grimoire/trie/beam-state.h"
#include "marisa/grimoire/trie/entry.h"

namespace marisa {
namespace grimoire {
namespace trie {

class Tail {
 public:
  Tail();

  void build(Vector<Entry> &entries, Vector<UInt32> *offsets,
      TailMode mode);

  void map(Mapper &mapper);
  void read(Reader &reader);
  void write(Writer &writer) const;

  __host__ __device__ void restore(Agent &agent, std::size_t offset) const;
  bool match(Agent &agent, std::size_t offset) const;
  __host__ __device__ bool prefix_match(Agent &agent, std::size_t offset) const;
  __device__ bool match(Label token, std::size_t &offset, bool &end) const;
  __device__ Label next_token(std::size_t &offset, bool &end) const;

  const Label &operator[](std::size_t offset) const {
    MARISA_DEBUG_IF(offset >= buf_.size(), MARISA_BOUND_ERROR);
    return buf_[offset];
  }

  TailMode mode() const {
    return end_flags_.empty() ? MARISA_TEXT_TAIL : MARISA_BINARY_TAIL;
  }

  bool empty() const {
    return buf_.empty();
  }
  std::size_t size() const {
    return buf_.size();
  }
  std::size_t total_size() const {
    return buf_.total_size() + end_flags_.total_size();
  }
  std::size_t io_size() const {
    return buf_.io_size() + end_flags_.io_size();
  }

  void clear();
  void swap(Tail &rhs);
  void prefetch() const;

 private:
  Vector<Label> buf_;
  BitVector end_flags_;

  void build_(Vector<Entry> &entries, Vector<UInt32> *offsets,
      TailMode mode);

  void map_(Mapper &mapper);
  void read_(Reader &reader);
  void write_(Writer &writer) const;

  // Disallows copy and assignment.
  Tail(const Tail &);
  Tail &operator=(const Tail &);
};

}  // namespace trie
}  // namespace grimoire
}  // namespace marisa

#endif  // MARISA_GRIMOIRE_TRIE_TAIL_H_
