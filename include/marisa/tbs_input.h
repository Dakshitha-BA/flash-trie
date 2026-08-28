#pragma once

#include <chrono>

namespace marisa {

struct TbsInput {
  const std::vector<UInt32> &beams;
  const std::vector<std::vector<Label>> &topk_id;
  const std::vector<std::vector<float>> &topk_logp;
  std::vector<std::vector<Label>> &out_sent;
  std::vector<double> &out_logp_norm;
  float token_logp_threshold;
  double sent_logp_threshold;
  double length_norm;
  bool early_exit;

  // Per-request GPU timing breakdown, in milliseconds.
  // Populated by tbs_gpu / tbs_gpu_batched when the library is built with
  // -DTBS_PROFILE. Left at 0 otherwise. The four phase times sum to kernel_ms.
  double t_expansion_ms = 0.0;      // child lookup (binary search / in-link)
  double t_validation_ms = 0.0;     // threshold + terminal checks + push
  double t_selection_ms = 0.0;      // top-B selection (sort / radix topk)
  double t_grid_overhead_ms = 0.0;  // grid sync barriers + setup/save/result
  double kernel_ms = 0.0;           // total device kernel time (sum of above)
};

inline double get_msec(
    std::chrono::time_point<std::chrono::high_resolution_clock> start,
    std::chrono::time_point<std::chrono::high_resolution_clock> end) {
  std::chrono::duration<double> elapsed = end - start;
  auto msec = 1000 * elapsed.count();
  return int(100 * msec) / 100.;  // truncate to 2 decimals
}

}  // namespace marisa
