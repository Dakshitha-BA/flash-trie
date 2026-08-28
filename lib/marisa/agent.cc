#include <new>

#include "marisa/agent.h"
#include "marisa/grimoire/trie.h"
#include "marisa/grimoire/cuda_allocator.h"

namespace marisa {

Agent::Agent(double length_norm)
    : query_(), key_(), state_(), length_norm_(length_norm) {}

Agent::Agent(const Agent &parent, Label token, float score) : Agent(parent.length_norm_) {
  // Copy query and add new token
  auto query_vec = parent.query_.vec();
  query_vec.push_back(token);
  Label* ptr = new Label[query_vec.size()];
  std::copy(query_vec.begin(), query_vec.end(), ptr);
  query_.set_str(ptr, query_vec.size());
  query_.set_id(parent.query().id());

  // Copy state
  init_state();
  auto& parent_key_buf = parent.state_->key_buf();
  state_->key_buf().resize(parent_key_buf.size());
  std::copy(
     parent_key_buf.begin(), parent_key_buf.end(), state_->key_buf().begin());

  auto& parent_history = parent.state_->history();
  assert(parent_history.empty());
  std::copy(
    parent_history.begin(), parent_history.end(), state_->history().begin());

  state_->set_node_id(parent.state_->node_id());
  state_->set_query_pos(parent.state_->query_pos());
  state_->set_status_code(parent.state_->status_code());
  state_->set_score(parent.state_->score() + score); // Add parent score
}

Agent::~Agent() {}

void Agent::set_query(const Label *ptr, std::size_t length) {
  MARISA_THROW_IF((ptr == NULL) && (length != 0), MARISA_NULL_ERROR);
  if (state_.get() != NULL) {
    state_->reset();
  }
  query_.set_str(ptr, length);
}

void Agent::set_query(std::size_t key_id) {
  if (state_.get() != NULL) {
    state_->reset();
  }
  query_.set_id(key_id);
}

void Agent::init_state() {
  MARISA_THROW_IF(state_.get() != NULL, MARISA_STATE_ERROR);
  state_.reset(new (std::nothrow) grimoire::State);
  MARISA_THROW_IF(state_.get() == NULL, MARISA_MEMORY_ERROR);
}

void Agent::clear() {
  Agent().swap(*this);
}

void Agent::swap(Agent &rhs) {
  query_.swap(rhs.query_);
  key_.swap(rhs.key_);
  state_.swap(rhs.state_);
  std::swap(length_norm_, rhs.length_norm_);
}

}  // namespace marisa
