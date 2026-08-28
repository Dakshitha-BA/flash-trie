#ifndef MARISA_AGENT_H_
#define MARISA_AGENT_H_

#include <cmath>

#include "marisa/grimoire/trie/state.h"
#include "marisa/key.h"
#include "marisa/query.h"

namespace marisa {
namespace grimoire {
namespace trie {

class State;

}  // namespace trie
}  // namespace grimoire

class Agent {
 public:
  Agent(double length_norm = 0);
  Agent(const Agent &parent, Label token, float score);
  ~Agent();

  __host__ __device__ const Query &query() const {
    return query_;
  }
  const Key &key() const {
    return key_;
  }

  void set_query(const std::vector<Label>& str) {
    set_query(str.data(), str.size());
  }
  void set_query(const Label *ptr, std::size_t length);
  void set_query(std::size_t key_id);

  __host__ __device__ const grimoire::trie::State &state() const {
    return *state_;
  }
  __host__ __device__ grimoire::trie::State &state() {
    return *state_;
  }

  void set_key(const std::vector<Label>& str) {
    set_key(str.data(), str.size());
  }
  void set_key(const Label *ptr, std::size_t length) {
    MARISA_DEBUG_IF((ptr == NULL) && (length != 0), MARISA_NULL_ERROR);
    MARISA_DEBUG_IF(length > MARISA_UINT32_MAX, MARISA_SIZE_ERROR);
    key_.set_str(ptr, length);
  }
  void set_key(std::size_t id) {
    MARISA_DEBUG_IF(id > MARISA_UINT32_MAX, MARISA_SIZE_ERROR);
    key_.set_id(id);
  }

  bool has_state() const {
    return state_.get() != NULL;
  }
  void init_state();

  void clear();
  void swap(Agent &rhs);

  __host__ __device__ double logp() const {
    return state_->score();
  }
  double logp_norm() const {
    return logp() *
           std::pow(6.0 / (5.0 + (double)query_.length()), length_norm_);
  }

 private:
  Query query_;
  Key key_;
  scoped_ptr<grimoire::trie::State> state_;
  double length_norm_;

  // Disallows copy and assignment.
  Agent(const Agent &);
  Agent &operator=(const Agent &);
};

}  // namespace marisa

#endif  // MARISA_AGENT_H_
