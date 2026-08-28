#include <cooperative_groups.h>
#include <nvtx3/nvToolsExt.h>
#include <omp.h>

#include <cub/block/block_scan.cuh>

#include "louds-trie.h"
#include "marisa/grimoire/cuda_check.h"

static constexpr int KERNEL_BLOCK_SIZE = 512;
#include "marisa/grimoire/trie/candidates-gpu.h"

#define CUDA_DEBUG(...)  // printf(__VA_ARGS__);

#ifdef TBS_PROFILE
// Raw clock-cycle counter indices recorded by the grid-leader thread
// (grid.thread_rank() == 0). See kNumPhaseCounters in beam-search-state.h for
// how these are combined on the host into the four reported phase times.
enum {
  PH_EXP = 0,    // leader expansion cycles (binary search / in-link lookup)
  PH_VAL = 1,    // leader validation cycles (threshold + terminal + push)
  PH_SEL = 2,    // selection wall span (trim regions incl. internal grid sync)
  PH_PROC = 3,   // process_outpos wall span incl. trailing grid-sync barrier
  PH_TOTAL = 4,  // total kernel wall span
};
#define PROF_IS_LEADER (blockIdx.x == 0 && threadIdx.x == 0)
#endif

namespace cg = cooperative_groups;

namespace marisa {
namespace grimoire {
namespace trie {

size_t get_in_buffer_size(size_t topk_len, size_t num_outpos) {
  // Format: outpos_beams, topk_id, topk_logp
  size_t outpos_beams_bytes = sizeof(UInt32) * num_outpos;
  size_t topk_id_bytes = sizeof(Label) * topk_len * num_outpos;
  size_t topk_logp_bytes = sizeof(float) * topk_len * num_outpos;

  size_t total_bytes = outpos_beams_bytes + topk_id_bytes + topk_logp_bytes;
  return total_bytes;
}

__device__ __host__ size_t get_out_buffer_size(
    size_t num_beams, size_t num_tokens) {
  // Format: total_bytes, num_beams, logp_norm, sent_offset, sent
  size_t logp_norm_bytes = sizeof(double) * num_beams;
  size_t sent_offset_bytes = sizeof(size_t) * num_beams;
  size_t sent_bytes = sizeof(Label) * num_tokens;

  size_t total_bytes =
      2 * sizeof(size_t) + logp_norm_bytes + sent_offset_bytes + sent_bytes;
  return total_bytes;
}

BeamSearchState::BeamSearchState(size_t max_num_topk, size_t max_num_outpos,
    size_t max_num_out_beams, size_t max_num_out_tokens,
    size_t max_num_inter_beams, size_t candidates_buffer_size,
    size_t radix_topk_threshold)
    : cur_(make_managed<CandidatesGPU>(
          candidates_buffer_size, radix_topk_threshold)),
      next_(make_managed<CandidatesGPU>(
          candidates_buffer_size, radix_topk_threshold)),
      selected_(make_managed<CandidatesGPU>(
          candidates_buffer_size, radix_topk_threshold)),
      max_num_topk_(max_num_topk), max_num_outpos_(max_num_outpos),
      max_num_out_beams_(max_num_out_beams),
      max_num_out_tokens_(max_num_out_tokens),
      max_num_inter_beams_(max_num_inter_beams) {
  cudaMalloc(&d_trie_, sizeof(LoudsTrie));

  cudaMalloc(&beams_, sizeof(BeamState) * max_num_inter_beams_);
  cudaMalloc(&next_beam_id_, sizeof(int));

  size_t in_buffer_size = get_in_buffer_size(max_num_topk_, max_num_outpos_);
  cudaMalloc(&in_buffer_, in_buffer_size);
  cudaMallocHost(&host_in_buffer_, in_buffer_size);

  size_t out_buffer_size =
      get_out_buffer_size(max_num_out_beams_, max_num_out_tokens_);
  cudaMalloc(&out_buffer_, out_buffer_size);
  cudaMallocHost(&host_out_buffer_, out_buffer_size);

  cudaStreamCreate(&stream_);
  setup_device_ptr_done_ = false;

  cudaMalloc(&phase_clocks_, sizeof(unsigned long long) * kNumPhaseCounters);
  sm_clock_khz_ = 0;
}

BeamSearchState::BeamSearchState(BeamSearchState&& rhs) noexcept
    : d_trie_(rhs.d_trie_), cur_(rhs.cur_), next_(rhs.next_),
      selected_(rhs.selected_), beams_(rhs.beams_),
      next_beam_id_(rhs.next_beam_id_), in_buffer_(rhs.in_buffer_),
      host_in_buffer_(rhs.host_in_buffer_), out_buffer_(rhs.out_buffer_),
      host_out_buffer_(rhs.host_out_buffer_), stream_(rhs.stream_),
      num_ctas_(rhs.num_ctas_),
      setup_device_ptr_done_(rhs.setup_device_ptr_done_),
      phase_clocks_(rhs.phase_clocks_), sm_clock_khz_(rhs.sm_clock_khz_),
      max_num_topk_(rhs.max_num_topk_), max_num_outpos_(rhs.max_num_outpos_),
      max_num_out_beams_(rhs.max_num_out_beams_),
      max_num_out_tokens_(rhs.max_num_out_tokens_),
      max_num_inter_beams_(rhs.max_num_inter_beams_) {
  // Null out source so its destructor doesn't free our resources
  rhs.d_trie_ = nullptr;
  rhs.cur_ = nullptr;
  rhs.next_ = nullptr;
  rhs.selected_ = nullptr;
  rhs.beams_ = nullptr;
  rhs.next_beam_id_ = nullptr;
  rhs.in_buffer_ = nullptr;
  rhs.host_in_buffer_ = nullptr;
  rhs.out_buffer_ = nullptr;
  rhs.host_out_buffer_ = nullptr;
  rhs.stream_ = nullptr;
  rhs.phase_clocks_ = nullptr;
}

void BeamSearchState::swap(BeamSearchState& rhs) {
  std::swap(d_trie_, rhs.d_trie_);
  std::swap(cur_, rhs.cur_);
  std::swap(next_, rhs.next_);
  std::swap(selected_, rhs.selected_);

  std::swap(beams_, rhs.beams_);
  std::swap(next_beam_id_, rhs.next_beam_id_);

  std::swap(in_buffer_, rhs.in_buffer_);
  std::swap(host_in_buffer_, rhs.host_in_buffer_);

  std::swap(out_buffer_, rhs.out_buffer_);
  std::swap(host_out_buffer_, rhs.host_out_buffer_);

  std::swap(stream_, rhs.stream_);
  std::swap(num_ctas_, rhs.num_ctas_);
  std::swap(setup_device_ptr_done_, rhs.setup_device_ptr_done_);
  std::swap(phase_clocks_, rhs.phase_clocks_);
  std::swap(sm_clock_khz_, rhs.sm_clock_khz_);

  std::swap(max_num_topk_, rhs.max_num_topk_);
  std::swap(max_num_outpos_, rhs.max_num_outpos_);
  std::swap(max_num_out_beams_, rhs.max_num_out_beams_);
  std::swap(max_num_out_tokens_, rhs.max_num_out_tokens_);
  std::swap(max_num_inter_beams_, rhs.max_num_inter_beams_);
}

BeamSearchState::~BeamSearchState() {
  cudaFree(d_trie_);
  for (auto ptr : {cur_, next_, selected_}) {
    cudaFree(ptr);
  }

  cudaFree(beams_);
  cudaFree(next_beam_id_);

  cudaFree(in_buffer_);
  cudaFreeHost(host_in_buffer_);

  cudaFree(out_buffer_);
  cudaFreeHost(host_out_buffer_);

  cudaFree(phase_clocks_);

  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

void BeamSearchState::prefetch() const {
  cudaMemLocation location;
  location.type = cudaMemLocationTypeDevice;
  location.id = 0;

  for (auto ptr : {cur_, next_, selected_}) {
    cudaMemPrefetchAsync(ptr, sizeof(CandidatesGPU), location, 0);
  }
}

void BeamSearchState::setup_device_ptr(const LoudsTrie* trie) {
  if (not setup_device_ptr_done_) {
    cudaMemcpyAsync(d_trie_, (const void*)trie, sizeof(LoudsTrie),
        cudaMemcpyHostToDevice, stream_);
    setup_device_ptr_done_ = true;
  }
}

BeamSearchArgs::BeamSearchArgs(BeamSearchState& state,
    double token_logp_threshold, double sent_logp_threshold, double length_norm,
    bool early_exit, size_t topk_len, size_t num_outpos)
    : trie_(state.d_trie_), cur_(state.cur_), next_(state.next_),
      selected_(state.selected_),

      beams_(state.beams_), max_beam_id_(state.max_num_inter_beams_),
      next_beam_id_(state.next_beam_id_),

      out_buffer_(state.out_buffer_), phase_clocks_(state.phase_clocks_),

      token_logp_threshold_(token_logp_threshold),
      sent_logp_threshold_(sent_logp_threshold), length_norm_(length_norm),
      early_exit_(early_exit), topk_len_(topk_len), num_outpos_(num_outpos) {
  char* buffer = (char*)state.in_buffer_;
  outpos_beams_ = (UInt32*)buffer;
  buffer += num_outpos * sizeof(UInt32);

  topk_id_ = (Label*)buffer;
  buffer += topk_len * num_outpos * sizeof(Label);

  topk_logp_ = (float*)buffer;
}

__device__ void LoudsTrie::extend_beam_binary_child_search(
    const BeamState& parent, const BeamSearchArgs& args) const {
  const size_t base_louds_pos = louds_.select0(parent.node_id_) + 1;
  const size_t base_node_id = base_louds_pos - parent.node_id_ - 1;

  const size_t end_louds_pos = louds_.warp_find_next_unset(base_louds_pos);
  const size_t degree = end_louds_pos - base_louds_pos;

  if (degree == 0) {
    return;
  }

  // Each thread binary searches a topk key in children list of parent
  for (size_t key_id = threadIdx.x; key_id < args.topk_len_;
       key_id += blockDim.x) {
#ifdef TBS_PROFILE
    const bool prof = PROF_IS_LEADER;
    unsigned long long ts = prof ? clock64() : 0;
#endif
    auto score = args.topk_logp_[key_id];
    bool low_score = score <= args.token_logp_threshold_ ||
                     parent.logp() + score <= args.sent_logp_threshold_;
    if (low_score) {
#ifdef TBS_PROFILE
      if (prof) args.phase_clocks_[PH_VAL] += clock64() - ts;
#endif
      continue;
    }
    auto target_token = args.topk_id_[key_id];
#ifdef TBS_PROFILE
    if (prof) {
      unsigned long long n = clock64();
      args.phase_clocks_[PH_VAL] += n - ts;
      ts = n;
    }
#endif

    // Result of binary search
    Label next_token = -1u;
    size_t cur_node_id = -1lu;
    size_t link_offset = -1lu;
    bool child_is_link = false;
    bool end_link = false;

    // Binary search
    int begin = 0, end = degree - 1;
    while (begin <= end) {
      auto mid = (begin + end) >> 1;

      cur_node_id = base_node_id + mid;
      child_is_link = link_flags_[cur_node_id];

      if (!child_is_link) {
        next_token = get_label(cur_node_id);
      } else {
        auto link_id = link_flags_.rank1(cur_node_id);
        link_offset = get_link(cur_node_id, link_id);
        next_token = tail_.next_token(link_offset, end_link);
      }

      if (next_token == target_token) {
        begin = mid;
        break;
      } else if (next_token < target_token) {
        begin = mid + 1;
      } else {
        end = mid - 1;
      }
    }  // begin <= end

#ifdef TBS_PROFILE
    if (prof) {
      unsigned long long n = clock64();
      args.phase_clocks_[PH_EXP] += n - ts;
      ts = n;
    }
#endif

    if (begin > end) {
#ifdef TBS_PROFILE
      if (prof) args.phase_clocks_[PH_VAL] += clock64() - ts;
#endif
      continue;  // no match
    }

    // create child beam from parent
    assert(next_token == target_token);
    auto child = BeamState(parent, next_token, score);
    child.node_id_ = cur_node_id;
    child.in_link_ = child_is_link;
    child.link_offset_ = link_offset;
    child.end_link_ = end_link;

    // update next and selected
    if (!child.in_link_) {
      if (args.trie_->is_terminal(child.node_id_)) {
        args.selected_->push(child);
      }
      args.next_->push(child);
    } else {
      if (child.end_link_) {
        if (args.trie_->is_terminal(child.node_id_)) {
          args.selected_->push(child);
        }
      } else {
        args.next_->push(child);
      }
    }  // in_link
#ifdef TBS_PROFILE
    if (prof) args.phase_clocks_[PH_VAL] += clock64() - ts;
#endif
  }  // topk index
}

__device__ void LoudsTrie::extend_beam_in_link(
    const BeamState& parent, const BeamSearchArgs& args) const {
#ifdef TBS_PROFILE
  const bool prof = PROF_IS_LEADER;
  unsigned long long ts = prof ? clock64() : 0;
#endif
  auto link_offset = parent.link_offset_;
  auto end_link = parent.end_link_;
  auto next_token = tail_.next_token(link_offset, end_link);
#ifdef TBS_PROFILE
  if (prof) {
    unsigned long long n = clock64();
    args.phase_clocks_[PH_EXP] += n - ts;
    ts = n;
  }
#endif

  // all threads search for next token in topk_id, with one potential match
  for (size_t index = threadIdx.x; index < args.topk_len_;
       index += blockDim.x) {
    bool match = args.topk_id_[index] == next_token;
    if (match) {
      auto score = args.topk_logp_[index];
      bool low_score = score <= args.token_logp_threshold_ ||
                       parent.logp() + score <= args.sent_logp_threshold_;
      if (low_score) {
        break;
      }

      auto child = BeamState(parent, next_token, score);
      child.link_offset_ = link_offset;
      child.end_link_ = end_link;

      if (child.end_link_) {
        if (args.trie_->is_terminal(child.node_id_)) {
          args.selected_->push(child);
        }
      } else {
        // Consume one link token and move to next outpos
        args.next_->push(child);
      }
      break;
    }  // match
  }  // topk index
#ifdef TBS_PROFILE
  if (prof) args.phase_clocks_[PH_VAL] += clock64() - ts;
#endif
}

__device__ void process_outpos(const BeamSearchArgs& args) {
  // For both link and non-link parents,
  // each parent beam is processed by one thread block
  for (size_t parent_beam_id = blockIdx.x; parent_beam_id < args.cur_->size();
       parent_beam_id += gridDim.x) {
    const auto& parent = args.cur_->at(parent_beam_id);
    if (parent.in_link_) {
      args.trie_->extend_beam_in_link(parent, args);
    }
  }

  for (size_t parent_beam_id = blockIdx.x; parent_beam_id < args.cur_->size();
       parent_beam_id += gridDim.x) {
    const auto& parent = args.cur_->at(parent_beam_id);
    if (!parent.in_link_) {
      args.trie_->extend_beam_binary_child_search(parent, args);
    }
  }
}

__device__ void generate_result(const BeamSearchArgs& args) {
  if (blockIdx.x == 0) {
    auto buffer = (char*)args.out_buffer_;
    auto total_bytes = (size_t*)buffer;
    buffer += sizeof(size_t);

    // copy num_beams
    auto& selected = args.selected_;
    const size_t num_beams = selected->size();
    if (threadIdx.x == 0) {
      *((size_t*)buffer) = num_beams;
    }
    buffer += sizeof(size_t);

    auto out_logp_norm = (double*)buffer;
    assert(((uint64_t)buffer) % sizeof(double) == 0);
    buffer += sizeof(double) * num_beams;

    // offset that marks start, end for a beam
    size_t* out_sent_offset = (size_t*)buffer;
    buffer += sizeof(size_t) * num_beams;
    // Copy logp norm and beam lengths
    for (size_t id = threadIdx.x; id < num_beams; id += blockDim.x) {
      // Output is in ascending score order. Selected is in descending order
      auto out_id = num_beams - 1 - id;
      auto beam = selected->at(id);
      out_sent_offset[out_id] = beam.depth_;
      out_logp_norm[out_id] = beam.logp_norm();
    }
    __syncthreads();

    // In-place inclusive sum of beam lengths using BlockScan
    size_t block_start_pos = 0;
    size_t block_cum_aggregate = 0;
    using BlockScan = cub::BlockScan<size_t, KERNEL_BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    while (block_start_pos < num_beams) {
      size_t out_id = block_start_pos + threadIdx.x;
      size_t thread_data = out_id < num_beams ? out_sent_offset[out_id] : 0;
      size_t block_aggregate;
      BlockScan(temp_storage)
          .InclusiveSum(thread_data, thread_data, block_aggregate);

      out_sent_offset[out_id] = thread_data + block_cum_aggregate;
      block_start_pos += blockDim.x;
      block_cum_aggregate += block_aggregate;
      __syncthreads();
    }

    // Copy sent tokens
    auto out_sent = (Label*)buffer;
    size_t num_sent = out_sent_offset[num_beams - 1];
    buffer += sizeof(Label) * num_sent;

    for (size_t id = threadIdx.x; id < num_beams; id += blockDim.x) {
      auto beam = selected->at(id);
      auto out_id = selected->size() - 1 - id;
      auto end_offset = out_sent_offset[out_id];
      size_t beam_length = beam.depth_;

      // Write in reverse order
      for (size_t pos = 0; pos < beam_length; pos++) {
        out_sent[end_offset - 1 - pos] = beam.token_;
        auto parent_beam_id = beam.parent_beam_id_;
        assert(parent_beam_id != -1lu);
        if (parent_beam_id == -1lu) {
          printf("backtrace failed for beam %lu length %lu pos %lu\n", id,
              beam_length, pos);
          break;
        }
        beam = args.beams_[parent_beam_id];
      }
    }

    // save total byte size of out_buffer
    if (threadIdx.x == 0) {
      *total_bytes = get_out_buffer_size(num_beams, num_sent);
    }
    __syncthreads();
  }
}

__global__ void __launch_bounds__(KERNEL_BLOCK_SIZE, 1)
    beam_search_kernel(BeamSearchArgs args) {
  cg::grid_group grid = cg::this_grid();

#ifdef TBS_PROFILE
  const bool prof = (grid.thread_rank() == 0);
  unsigned long long k_start = 0;
  unsigned long long sel_ts = 0;
  unsigned long long proc_ts = 0;
  if (prof) {
    for (int i = 0; i < kNumPhaseCounters; i++) args.phase_clocks_[i] = 0;
    k_start = clock64();
  }
#endif

  size_t final_beam = args.outpos_beams_[args.num_outpos_ - 1];
  if (grid.thread_rank() == 0) {
    args.cur_->clear();
    args.next_->clear();
    args.selected_->clear();

    auto& root = args.beams_[0];
    *(args.next_beam_id_) = 1;
    root.beam_id_ = 0;
    root.length_norm_ = args.length_norm_;

    args.cur_->set_capacity(1);
    args.cur_->push(root);

    args.selected_->set_capacity(final_beam);
    args.next_->set_capacity(args.outpos_beams_[0]);
  }
  cg::sync(grid);

  for (size_t outpos = 0; outpos < args.num_outpos_; outpos++) {
    if (grid.thread_rank() == 0) {
      CUDA_DEBUG("outpos %lu cur %lu selected %lu", outpos, args.cur_->size(),
          args.selected_->size());
    }

    if (args.cur_->empty() ||
        (args.early_exit_ && args.selected_->size() >= final_beam)) {
      if (grid.thread_rank() == 0) {
        CUDA_DEBUG("\n");
      }
      break;
    }

#ifdef TBS_PROFILE
    if (prof) proc_ts = clock64();
#endif
    process_outpos(args);
    cg::sync(grid);
#ifdef TBS_PROFILE
    // Wall span of the expand+validate phase across the whole grid, including
    // the barrier wait for the slowest block (true critical-path time).
    if (prof) args.phase_clocks_[PH_PROC] += clock64() - proc_ts;
#endif

    if (grid.thread_rank() == 0) {
      CUDA_DEBUG(" next %lu selected %lu\n", args.next_->size(),
          args.selected_->size());
    }
#ifdef TBS_PROFILE
    if (prof) sel_ts = clock64();
#endif
    args.next_->trim(false);
#ifdef TBS_PROFILE
    if (prof) args.phase_clocks_[PH_SEL] += clock64() - sel_ts;
#endif
    args.next_->save(args.beams_, args.next_beam_id_, args.max_beam_id_);

    if (grid.thread_rank() == 0) {
      // Prepare args.next_ (args.cur_ before swap) for next outpos
      if (outpos + 1 < args.num_outpos_) {
        args.cur_->clear();
        args.cur_->set_capacity(args.outpos_beams_[outpos + 1]);
      }
    }
    cg::sync(grid);

    if (args.selected_->size() > args.selected_->buffer_size() / 4) {
      // Proactively trim to avoid overflowing internal buffer of selected_
#ifdef TBS_PROFILE
      if (prof) sel_ts = clock64();
#endif
      args.selected_->trim(false);
#ifdef TBS_PROFILE
      if (prof) args.phase_clocks_[PH_SEL] += clock64() - sel_ts;
#endif
      cg::sync(grid);
    }

    // swap cur, next
    CandidatesGPU* temp = args.cur_;
    args.cur_ = args.next_;
    args.next_ = temp;

    args.topk_id_ = args.topk_id_ + args.topk_len_;
    args.topk_logp_ = args.topk_logp_ + args.topk_len_;
  }

#ifdef TBS_PROFILE
  if (prof) sel_ts = clock64();
#endif
  args.selected_->trim(true);
#ifdef TBS_PROFILE
  if (prof) args.phase_clocks_[PH_SEL] += clock64() - sel_ts;
#endif
  cg::sync(grid);

  generate_result(args);

#ifdef TBS_PROFILE
  // Record the total kernel wall span; the host derives the four phase times
  // from PH_PROC, PH_SEL and this total (see copy_output_to_cpu).
  if (prof) args.phase_clocks_[PH_TOTAL] = clock64() - k_start;
#endif
}

void copy_input_to_gpu(const TbsInput& input, BeamSearchState& bs_state);
void copy_output_to_cpu(TbsInput& input, const BeamSearchState& bs_state);

void LoudsTrie::init_tbs_gpu(size_t max_batch_size, size_t max_num_topk,
    size_t max_num_outpos, size_t max_num_out_beams, size_t max_num_out_tokens,
    size_t max_num_inter_beams, size_t radix_topk_threshold) {
  assert(max_batch_size <= kMaxBatchSizeLimit);
  assert(radix_topk_threshold <= CandidatesGPU::max_sort_topk_capacity_);
  max_batch_size_ = max_batch_size;
  bs_state_.clear();
  bs_state_.reserve(max_batch_size_);
  // Size the buffer to hold one full step of candidates:
  // each of max_num_out_beams active beams can emit up to max_num_topk
  // children.
  size_t candidates_buffer_size = max_num_out_beams * max_num_topk;
  for (size_t i = 0; i < max_batch_size_; i++) {
    bs_state_.emplace_back(max_num_topk, max_num_outpos, max_num_out_beams,
        max_num_out_tokens, max_num_inter_beams, candidates_buffer_size,
        radix_topk_threshold);
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  size_t num_sms = deviceProp.multiProcessorCount;
  assert(num_sms >= max_batch_size_);

  // Cooperative launches require every block to be co-resident, so the total
  // number of blocks across all concurrently launched batches must not exceed
  // maxActiveBlocksPerSM * num_sms for this kernel. Query the real occupancy
  // (it can drop when extra instrumentation raises register pressure) and cap
  // the per-batch grid so the sum stays within that bound.
  int max_blocks_per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks_per_sm, (void*)beam_search_kernel, KERNEL_BLOCK_SIZE, 0);
  if (max_blocks_per_sm < 1) max_blocks_per_sm = 1;
  size_t max_total_ctas = (size_t)max_blocks_per_sm * num_sms;
  // Blocks available to each batch when they run concurrently.
  size_t ctas_per_batch = max_total_ctas / max_batch_size_;
  if (ctas_per_batch < 1) ctas_per_batch = 1;

  for (size_t batch_id = 0; batch_id < max_batch_size_; batch_id++) {
    uint32_t num_ctas = (uint32_t)(num_sms / max_batch_size_);
    if (batch_id < (num_sms % max_batch_size_)) {
      num_ctas += 1;
    }
    if (num_ctas > ctas_per_batch) {
      num_ctas = (uint32_t)ctas_per_batch;
    }
    if (num_ctas < 1) num_ctas = 1;
    bs_state_[batch_id].num_ctas_ = num_ctas;
    bs_state_[batch_id].sm_clock_khz_ = deviceProp.clockRate;
  }
}

void LoudsTrie::tbs_gpu_launch_async(TbsInput& input, size_t batch_id) const {
  assert(max_batch_size_ > 0 and batch_id < max_batch_size_);

  auto& bs_state = bs_state_[batch_id];

  auto topk_len = input.topk_id[0].size();
  auto num_outpos = input.beams.size();
  assert(topk_len <= bs_state.max_num_topk_);
  assert(num_outpos <= bs_state.max_num_outpos_);
  bs_state.setup_device_ptr(this);
  copy_input_to_gpu(input, bs_state);

  BeamSearchArgs args(bs_state, input.token_logp_threshold,
      input.sent_logp_threshold, input.length_norm, input.early_exit, topk_len,
      num_outpos);

  dim3 dimGrid(bs_state.num_ctas_, 1, 1);
  dim3 dimBlock(KERNEL_BLOCK_SIZE, 1, 1);
  void* kernel_args[] = {(void*)&args};
  CUDA_CHECK(cudaLaunchCooperativeKernel((void*)beam_search_kernel, dimGrid,
      dimBlock, kernel_args, 0, bs_state.stream_));
}

void LoudsTrie::tbs_gpu(TbsInput& input, size_t batch_id) const {
  nvtxRangePushA("tbs_gpu");
  auto final_beam = input.beams[input.beams.size() - 1];
  if (final_beam == 0) {
    return;
  }
  assert(final_beam <= bs_state_[batch_id].max_num_out_beams_);

  tbs_gpu_launch_async(input, batch_id);
  copy_output_to_cpu(input, bs_state_[batch_id]);
  nvtxRangePop();
}

void LoudsTrie::tbs_gpu_batched(std::vector<TbsInput>& inputs) const {
  nvtxRangePushA("tbs_gpu_batched");
  const auto batch_size = inputs.size();
#pragma omp parallel for schedule(static, 1)
  for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
    auto& input = inputs[batch_id];
    auto final_beam = input.beams[input.beams.size() - 1];
    assert(final_beam <= bs_state_[batch_id].max_num_out_beams_);
    if (final_beam != 0) {
      tbs_gpu_launch_async(input, batch_id);
      copy_output_to_cpu(input, bs_state_[batch_id]);
    }
  }
  nvtxRangePop();
}

void copy_input_to_gpu(const TbsInput& input, BeamSearchState& bs_state) {
  auto topk_len = input.topk_id[0].size();
  auto num_outpos = input.beams.size();

  char* buffer = (char*)bs_state.host_in_buffer_;

  size_t expected_bytes = get_in_buffer_size(topk_len, num_outpos);
  size_t consumed_bytes = 0;
  auto advance_buffer = [&](size_t num_elems, size_t elem_size) {
    size_t bytes = num_elems * elem_size;
    buffer += bytes;
    consumed_bytes += bytes;
  };

  // Format: outpos_beams, topk_id, topk_logp
  auto beams_buffer = (UInt32*)buffer;
  std::copy(input.beams.begin(), input.beams.end(), beams_buffer);
  advance_buffer(num_outpos, sizeof(UInt32));

  auto topk_id_buffer = (Label*)buffer;
  for (size_t outpos = 0; outpos < num_outpos; outpos++) {
    size_t offset = topk_len * outpos;
    std::copy(input.topk_id[outpos].begin(), input.topk_id[outpos].end(),
        topk_id_buffer + offset);
  }
  advance_buffer(topk_len * num_outpos, sizeof(Label));

  auto topk_logp_buffer = (float*)buffer;
  for (size_t outpos = 0; outpos < num_outpos; outpos++) {
    size_t offset = topk_len * outpos;
    std::copy(input.topk_logp[outpos].begin(), input.topk_logp[outpos].end(),
        topk_logp_buffer + offset);
  }
  advance_buffer(topk_len * num_outpos, sizeof(float));

  assert(consumed_bytes == expected_bytes);
  cudaMemcpyAsync(bs_state.in_buffer_, bs_state.host_in_buffer_, expected_bytes,
      cudaMemcpyHostToDevice, bs_state.stream_);
}

void copy_output_to_cpu(TbsInput& input, const BeamSearchState& bs_state) {
  auto num_outpos = input.beams.size();
  auto final_beam = input.beams[num_outpos - 1];

  size_t expected_bytes =
      get_out_buffer_size(final_beam, final_beam * num_outpos);
  cudaMemcpyAsync(bs_state.host_out_buffer_, bs_state.out_buffer_,
      expected_bytes, cudaMemcpyDeviceToHost, bs_state.stream_);
  CUDA_CHECK(cudaStreamSynchronize(bs_state.stream_));

#ifdef TBS_PROFILE
  {
    unsigned long long h[kNumPhaseCounters] = {0};
    cudaMemcpy(h, bs_state.phase_clocks_, sizeof(h), cudaMemcpyDeviceToHost);
    double khz = (double)bs_state.sm_clock_khz_;
    auto to_ms = [&](double c) {
      return khz > 0.0 ? c / khz : 0.0;  // cycles / kHz == ms
    };

    // Raw spans (cycles): leader exp/val (for ratio only), selection wall,
    // process wall (incl. barrier wait), and total kernel wall.
    double exp_c = (double)h[PH_EXP];
    double val_c = (double)h[PH_VAL];
    double sel_c = (double)h[PH_SEL];
    double proc_c = (double)h[PH_PROC];
    double total_c = (double)h[PH_TOTAL];

    // Split the process-phase wall time into expansion vs validation using the
    // leader thread's measured compute ratio (the only available signal for
    // their relative cost). This attributes load-imbalance barrier wait to the
    // real work phase rather than to overhead.
    double ev = exp_c + val_c;
    double exp_frac = ev > 0.0 ? exp_c / ev : 0.5;
    input.t_expansion_ms = to_ms(proc_c * exp_frac);
    input.t_validation_ms = to_ms(proc_c * (1.0 - exp_frac));
    input.t_selection_ms = to_ms(sel_c);

    // True overhead = total minus the expand/validate and selection wall spans
    // (grid-sync barriers outside those regions, setup, save, result gen,
    // buffer swaps, and host kernel launch is measured separately on the host).
    double ovh_c = total_c - proc_c - sel_c;
    if (ovh_c < 0.0) ovh_c = 0.0;
    input.t_grid_overhead_ms = to_ms(ovh_c);

    input.kernel_ms = to_ms(total_c);
  }
#endif

  char* buffer = (char*)bs_state.host_out_buffer_;
  size_t consumed_bytes = 0;
  auto advance_buffer = [&](size_t num_elems, size_t elem_size) {
    size_t bytes = num_elems * elem_size;
    buffer += bytes;
    consumed_bytes += bytes;
  };

  size_t total_bytes = *((size_t*)buffer);
  assert(total_bytes <= expected_bytes);
  advance_buffer(1, sizeof(size_t));

  size_t num_beams = *((size_t*)buffer);
  assert(num_beams <= final_beam);
  advance_buffer(1, sizeof(size_t));

  auto out_logp_buffer = (double*)buffer;
  advance_buffer(num_beams, sizeof(double));

  input.out_logp_norm.resize(num_beams);
  std::copy(
      out_logp_buffer, out_logp_buffer + num_beams, input.out_logp_norm.data());

  auto out_sent_offset = (size_t*)buffer;
  advance_buffer(num_beams, sizeof(size_t));
  size_t total_tokens = out_sent_offset[num_beams - 1];
  assert(total_tokens <= final_beam * num_outpos);
  assert(total_tokens <= bs_state.max_num_out_tokens_);

  auto out_sent_buffer = (Label*)buffer;
  advance_buffer(total_tokens, sizeof(Label));
  input.out_sent.resize(num_beams);

  for (size_t beam_id = 0; beam_id < num_beams; beam_id++) {
    size_t start_offset = beam_id == 0 ? 0 : out_sent_offset[beam_id - 1];
    size_t end_offset = out_sent_offset[beam_id];

    auto beam_tokens = end_offset - start_offset;
    input.out_sent[beam_id].resize(beam_tokens);

    std::copy(out_sent_buffer + start_offset, out_sent_buffer + end_offset,
        input.out_sent[beam_id].begin());
  }

  assert(total_bytes == consumed_bytes);
}

}  // namespace trie
}  // namespace grimoire
}  // namespace marisa
