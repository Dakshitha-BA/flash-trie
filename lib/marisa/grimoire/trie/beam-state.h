#pragma once

namespace marisa {
namespace grimoire {
namespace trie {

struct BeamState {
  __host__ __device__ BeamState(double length_norm = 1.0)
      : node_id_(0), token_(0), depth_(0), beam_id_(-1u), parent_beam_id_(-1u),
        score_(0), length_norm_(length_norm), link_offset_(-1lu),
        in_link_(false), end_link_(false) {}

  __host__ __device__ BeamState(
      const BeamState& parent, Label token, float score)
      : node_id_(parent.node_id_), token_(token), depth_(parent.depth_ + 1),
        beam_id_(-1u), parent_beam_id_(parent.beam_id_),
        score_(parent.score_ + score), length_norm_(parent.length_norm_),
        link_offset_(parent.link_offset_), in_link_(parent.in_link_),
        end_link_(parent.end_link_) {}

  __host__ __device__ double logp() const {
    return score_;
  }
  __host__ __device__ double logp_norm() const {
    return logp() * std::pow(6.0 / (5.0 + (double)depth_), length_norm_);
  }

  size_t node_id_;

  Label token_;
  uint32_t depth_;

  // Indices into buffer with all beams. Used for backtracing result beam.
  uint32_t beam_id_;
  uint32_t parent_beam_id_;

  double score_;
  double length_norm_;

  size_t link_offset_;
  bool in_link_;
  bool end_link_;

  static constexpr double min_score = -1000 * 1000.0;
};

class BeamStateComparer {
 public:
  __host__ __device__ bool operator()(
      const BeamState& l, const BeamState& r) const {
    return l.logp_norm() > r.logp_norm();
  }
};

}  // namespace trie
}  // namespace grimoire
}  // namespace marisa
