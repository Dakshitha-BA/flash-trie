#pragma once

#include <memory>
#include <mutex>
#include <queue>

#include "marisa/agent.h"
#include "marisa/grimoire/trie/state.h"

namespace marisa {
typedef std::shared_ptr<Agent> AgentPtr;

namespace grimoire {
namespace trie {

class AgentComparer {
 public:
  bool operator()(Agent* l, Agent* r) const {
    if (l->logp_norm() == r->logp_norm()) {
      // Break ties deterministically
      return l->state().node_id() > r->state().node_id();
    } else {
      return l->logp_norm() > r->logp_norm();
    }
  }

  bool operator()(AgentPtr l, AgentPtr r) const {
    return (*this)(l.get(), r.get());
  }
};

class Candidates {
 public:
  Candidates(unsigned int max)
      : cand_(), max_(max), pmtx_(std::make_shared<std::mutex>()) {}

  Candidates(const Candidates& c)
      : cand_(c.cand_), max_(c.max_), pmtx_(c.pmtx_) {}

  Candidates& operator=(const Candidates& other) {
    cand_ = other.cand_;
    max_ = other.max_;
    pmtx_ = other.pmtx_;
    return *this;
  }

  void push(AgentPtr a) {
    pmtx_->lock();

    if (cand_.size() < max_ || a->logp_norm() > cand_.top()->logp_norm())
      cand_.push(a);
    if (cand_.size() > max_) cand_.pop();

    pmtx_->unlock();
  }

  AgentPtr pop() {
    auto ret = cand_.top();
    cand_.pop();
    return ret;
  }

  bool empty() const {
    return cand_.empty();
  }

  std::size_t size() const {
    return cand_.size();
  }

  std::priority_queue<AgentPtr, std::vector<AgentPtr>, AgentComparer>&
  get_queue() {
    return cand_;
  }

 private:
  std::priority_queue<AgentPtr, std::vector<AgentPtr>, AgentComparer> cand_;
  unsigned int max_;
  std::shared_ptr<std::mutex> pmtx_;
};

}  // namespace trie
}  // namespace grimoire
}  // namespace marisa
