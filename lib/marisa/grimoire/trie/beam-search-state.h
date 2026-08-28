#pragma once

#include "marisa/grimoire/trie/beam-state.h"

namespace marisa {
namespace grimoire {
namespace trie {

class LoudsTrie;
struct CandidatesGPU;

// Number of raw clock-cycle counters recorded by the beam search kernel.
// Raw counters (set in the kernel):
//   0 = leader expansion cycles   (used only for the expansion:validation ratio)
//   1 = leader validation cycles  (used only for the ratio)
//   2 = selection wall span       (trim regions, incl. their internal grid sync)
//   3 = process wall span         (process_outpos + following grid sync;
//                                   captures the slowest block + barrier wait)
//   4 = total kernel wall span
// The host turns these into the four reported phase times:
//   expansion   = process_span * exp/(exp+val)
//   validation  = process_span * val/(exp+val)
//   selection   = selection_span
//   overhead    = total - process_span - selection_span  (true sync/launch/setup)
static constexpr int kNumPhaseCounters = 5;

struct BeamSearchState {
  BeamSearchState(size_t max_num_topk, size_t max_num_outpos,
      size_t max_num_out_beams, size_t max_num_out_tokens,
      size_t max_num_inter_beams, size_t candidates_buffer_size,
      size_t radix_topk_threshold);
  BeamSearchState(BeamSearchState &&rhs) noexcept;
  BeamSearchState(const BeamSearchState &rhs) = delete;
  BeamSearchState &operator=(const BeamSearchState &rhs) = delete;
  ~BeamSearchState();

  void swap(BeamSearchState &rhs);
  void prefetch() const;
  void setup_device_ptr(const LoudsTrie *trie);

  LoudsTrie *d_trie_;  // Device copy of LoudsTrie instance

  CandidatesGPU *cur_;
  CandidatesGPU *next_;
  CandidatesGPU *selected_;

  // Any state in frontier (cur) is stored in beams_ to generate result.
  BeamState *beams_;
  int *next_beam_id_;

  // Single buffer that comprises all input arrays and metadata.
  void *in_buffer_;
  void *host_in_buffer_;

  // Single buffer that comprises all output arrays and metadata.
  // Result can be transferred to CPU with just one D->H transfer.
  void *out_buffer_;
  void *host_out_buffer_;

  cudaStream_t stream_;
  uint32_t num_ctas_;
  bool setup_device_ptr_done_;

  // Device buffer of per-phase clock-cycle accumulators (kNumPhaseCounters)
  // written by the kernel when built with -DTBS_PROFILE, plus the SM clock
  // rate (kHz) used to convert cycles to milliseconds.
  unsigned long long *phase_clocks_;
  int sm_clock_khz_;

  size_t max_num_topk_;
  size_t max_num_outpos_;
  size_t max_num_out_beams_;
  size_t max_num_out_tokens_;  // Combined token count across all output beams
  // limit on sum of active beams across all output positions
  size_t max_num_inter_beams_;
};

// Kernel args
struct BeamSearchArgs {
  const LoudsTrie *trie_;

  CandidatesGPU *cur_;
  CandidatesGPU *next_;
  CandidatesGPU *selected_;

  BeamState *beams_;
  const size_t max_beam_id_;
  int *next_beam_id_;

  // Inputs
  const UInt32 *outpos_beams_;
  const Label *topk_id_;
  const float *topk_logp_;

  void *out_buffer_;

  // Per-phase clock-cycle accumulators (kNumPhaseCounters); see BeamSearchState.
  unsigned long long *phase_clocks_;

  // Place fields that are not buffers here
  const double token_logp_threshold_;
  const double sent_logp_threshold_;
  const double length_norm_;
  const bool early_exit_;

  const size_t topk_len_;
  const size_t num_outpos_;

  BeamSearchArgs(BeamSearchState &state, double token_logp_threshold,
      double sent_logp_threshold, double length_norm, bool early_exit,
      size_t topk_len, size_t num_outpos);
};

}  // namespace trie
}  // namespace grimoire
}  // namespace marisa
