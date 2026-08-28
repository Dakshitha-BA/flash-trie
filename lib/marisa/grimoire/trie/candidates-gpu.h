#pragma once

#include <cooperative_groups.h>

#include <algorithm>
#include <cassert>
#include <cub/block/block_load.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>

#include "candidates.h"
#include "marisa/grimoire/trie/beam-state.h"
#include "marisa/grimoire/trie/state.h"

namespace cg = cooperative_groups;

namespace marisa {
namespace grimoire {
namespace trie {

class CandidatesGPU {
 public:
  CandidatesGPU(size_t buffer_size, size_t radix_topk_threshold,
      unsigned int capacity = 0)
      : size_(0), buffer_size_(buffer_size),
        radix_topk_threshold_(radix_topk_threshold) {
    set_capacity(capacity);
    cudaMalloc(&cand_, sizeof(BeamState) * buffer_size_);
    cudaMalloc(&topk_cand_, sizeof(BeamState) * max_capacity_);
    cudaMalloc(&sort_state_, sizeof(BeamSortState) * buffer_size_);
    cudaMalloc(&sort_state_second_, sizeof(BeamSortState) * buffer_size_);
  }

  ~CandidatesGPU() {
    cudaFree(cand_);
    cudaFree(topk_cand_);
    cudaFree(sort_state_);
    cudaFree(sort_state_second_);
  }

  void swap(CandidatesGPU& rhs) {
    std::swap(cand_, rhs.cand_);
    std::swap(topk_cand_, rhs.topk_cand_);
    std::swap(sort_state_, rhs.sort_state_);
    std::swap(sort_state_second_, rhs.sort_state_second_);
    std::swap(capacity_, rhs.capacity_);
    std::swap(size_, rhs.size_);
  }

  __device__ void push(BeamState a) {
    auto pos = atomicAdd(&size_, 1u);
    assert(pos < buffer_size_);
    cand_[pos] = a;
    sort_state_[pos] = BeamSortState{(float)a.logp_norm(), pos};
  }

  __device__ BeamState operator[](std::size_t pos) const {
    assert(pos < capacity_);
    return cand_[pos];
  }

  __device__ BeamState at(std::size_t pos) const {
    assert(pos < capacity_);
    return cand_[pos];
  }

  __device__ void clear() {
    size_ = 0;
  }

  __device__ void trim(bool force_sort) {
    cg::grid_group grid = cg::this_grid();
    if (size_ > capacity_ or force_sort) {
#ifdef CAND_DEBUG
      if (threadIdx.x == 0 and blockIdx.x == 0) {
        printf("trimming candidates, size %u capacity %lu\n", size_, capacity_);
      }
#endif

      if (capacity_ <= radix_topk_threshold_) {
        sort_topk();
      } else {
        if (size_ > capacity_) {
          radix_topk();
        }
      }

      if (grid.thread_rank()) {
        size_ = min((size_t)size_, capacity_);
      }
    }
  }

  __device__ void save(BeamState* beams, int* next_beam_id, size_t max_beam_id);

  __host__ __device__ void set_capacity(std::size_t capacity) {
    assert(capacity <= max_capacity_);
    capacity_ = capacity;
  }

  __device__ std::size_t capacity() const {
    return capacity_;
  }

  __device__ bool empty() const {
    return size_ == 0;
  }

  __device__ std::size_t size() const {
    return size_;
  }

  __device__ std::size_t buffer_size() const {
    return buffer_size_;
  }

  static constexpr std::size_t max_sort_topk_capacity_ = 1024;

 private:
  // Metadata needed for sorting.
  struct BeamSortState {
    float logp_norm_;  // float to minimize sizeof(BeamSortState);
    uint32_t pos_;
  };

  BeamState* cand_;       // Buffer with all inserted elements
  BeamState* topk_cand_;  // Intermediate buffers with top-k sorted elements
  BeamSortState* sort_state_;  // Metadata of cand_ elements used in sorting
  BeamSortState* sort_state_second_;  // Double buffering for sort_state_

  std::size_t capacity_;
  unsigned int size_;
  std::size_t buffer_size_;
  const size_t radix_topk_threshold_;

  static constexpr std::size_t max_capacity_ = 8192;

  __device__ void sort_topk() {
    if (capacity_ <= 256) {
      sort_topk_executor<256>();
    } else if (capacity_ <= 512) {
      sort_topk_executor<512>();
    } else if (capacity_ <= 768) {
      sort_topk_executor<768>();
    } else if (capacity_ <= 1024) {
      sort_topk_executor<1024>();
    } else {
      if (threadIdx.x == 0 and blockIdx.x == 0) {
        printf("Cannot use sort when K > %zu, use radix topk instead\n",
            max_sort_topk_capacity_);
        assert(false);
      }
    }
  }

  template <size_t capacity>
  __device__ void sort_topk_executor();
  __device__ void radix_topk();

  // Disallows copy and assignment.
  CandidatesGPU(const CandidatesGPU&);
  CandidatesGPU& operator=(const CandidatesGPU&);
};

__device__ void CandidatesGPU::radix_topk() {
  cg::grid_group grid = cg::this_grid();

  static constexpr int bits_per_step = 8;
  static constexpr int total_bits = sizeof(float) * 8;
  assert(total_bits % bits_per_step == 0);
  static constexpr int num_steps = total_bits / bits_per_step;
  static constexpr int num_bins = 1 << bits_per_step;

  size_t rem_n = size_;
  size_t rem_k = capacity_;

  __shared__ uint32_t histogram[num_bins];

  __shared__ int topk_cand_insert_pos;
  __shared__ int sort_state_insert_pos;

  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      topk_cand_insert_pos = 0;
      sort_state_insert_pos = 0;
    }

    for (int step = 0; step < num_steps; step++) {
#ifdef CAND_DEBUG
      if (threadIdx.x == 0) {
        printf("Step %d rem_n %lu rem_k %lu\n", step, rem_n, rem_k);
      }
#endif
      if (rem_n == 0) {
        break;
      }

      // Clear histogram
      for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        histogram[bin] = 0;
      }
      __syncthreads();

      cub::BFEDigitExtractor<unsigned> extractor(
          total_bits - (step + 1) * bits_per_step, bits_per_step);
      auto compute_bin = [&](size_t pos) {
        // all values are negative floats, so we safely cast to uint32_t
        auto score = static_cast<uint32_t>(-sort_state_[pos].logp_norm_);
        uint32_t bin = extractor.Digit(score);
        assert(bin < num_bins);
        return bin;
      };

      // Compute histogram
      for (int pos = threadIdx.x; pos < rem_n; pos += blockDim.x) {
        auto bin = compute_bin(pos);
        atomicAdd(&histogram[bin], 1u);
      }
      __syncthreads();

      // Inclusive prefix sum of histogram
      if (threadIdx.x == 0) {
        for (int bin = 1; bin < num_bins; bin++) {
          histogram[bin] = histogram[bin - 1] + histogram[bin];
        }
      }
      __syncthreads();

      // Find pivot_bin
      int pivot_bin = 0;
      while (pivot_bin < num_bins and histogram[pivot_bin] < rem_k) {
        pivot_bin++;
      }

      if (threadIdx.x == 0) {
        assert(pivot_bin <= num_bins);

#ifdef CAND_DEBUG
        printf("Pivot bin %d\n", pivot_bin);
        for (int bin = 0; bin < num_bins; bin++) {
          printf(" %u", histogram[bin]);
        }
        printf("\n");
#endif
      }

      if (pivot_bin == 0) {
        continue;
      }

      // Copy elements whose bin < pivot_bin into topk_cand
      for (int pos = threadIdx.x; pos < rem_n; pos += blockDim.x) {
        auto bin = compute_bin(pos);
        if (bin < pivot_bin) {
          auto unsorted_pos = sort_state_[pos].pos_;
          auto insert_pos = atomicAdd(&topk_cand_insert_pos, 1);
          topk_cand_[insert_pos] = cand_[unsorted_pos];
        }
      }

      // Copy pivot_bin elements to front of sort_state_second
      for (int pos = threadIdx.x; pos < rem_n; pos += blockDim.x) {
        auto bin = compute_bin(pos);
        if (bin == pivot_bin) {
          auto insert_pos = atomicAdd(&sort_state_insert_pos, 1);
          sort_state_second_[insert_pos] = sort_state_[pos];
        }
      }

      // Adjust rem_k, rem_n
      if (threadIdx.x == 0) {
        assert(rem_k >= histogram[pivot_bin - 1]);
      }
      rem_k -= histogram[pivot_bin - 1];
      rem_n = histogram[pivot_bin];

      if (step == num_steps - 1 && rem_k > 0) {
        // In last step, copy rem_k elements of pivot_bin into topk_cand
        for (int pos = threadIdx.x; pos < rem_n; pos += blockDim.x) {
          auto bin = compute_bin(pos);
          if (bin == pivot_bin) {
            auto unsorted_pos = sort_state_[pos].pos_;
            auto insert_pos = atomicAdd(&topk_cand_insert_pos, 1);
            if (insert_pos < capacity_) {
              topk_cand_[insert_pos] = cand_[unsorted_pos];
            } else {
              break;
            }
          }
        }
      }
      __syncthreads();

      // Reset for next step
      if (threadIdx.x == 0) {
        auto temp = sort_state_;
        sort_state_ = sort_state_second_;
        sort_state_second_ = temp;

        sort_state_insert_pos = 0;
      }

      __syncthreads();
    }

#ifdef CAND_DEBUG
    if (threadIdx.x == 0) {
      for (size_t pos = 0; pos < min((size_t)size_, capacity_); pos++) {
        printf(" %f", topk_cand_[pos].logp_norm());
      }
      printf("\n");
    }
#endif
  }
  cg::sync(grid);
}

template <size_t capacity>
__device__ void CandidatesGPU::sort_topk_executor() {
  cg::grid_group grid = cg::this_grid();
  constexpr size_t elems_per_block = 2 * capacity;
  constexpr size_t elems_per_thread = elems_per_block / KERNEL_BLOCK_SIZE;

  if (size_ <= 1) {
    return;
  }
  size_t remaining_blocks = (size_ - 1) / elems_per_block + 1;

  while (remaining_blocks >= 1) {
    // 1. sort current set of blocks
    for (size_t block_id = blockIdx.x; block_id < remaining_blocks;
         block_id += gridDim.x) {
      size_t block_offset = block_id * elems_per_block;

      // Load from global mem to registers
      BeamSortState thread_data[elems_per_thread];
      for (size_t src = block_offset + threadIdx.x, dst = 0;
           dst < elems_per_thread; src += blockDim.x, dst++) {
        if (src < size_) {
          thread_data[dst] = sort_state_[src];
        } else {
          thread_data[dst] = BeamSortState{BeamState::min_score, 0};
        }
      }
      __syncthreads();

      // Sort block
      class BeamSortStateComparer {
       public:
        __device__ bool operator()(
            const BeamSortState& l, const BeamSortState& r) const {
          return l.logp_norm_ > r.logp_norm_;
        }
      };

      using BlockMergeSort = cub::BlockMergeSort<BeamSortState,
          KERNEL_BLOCK_SIZE, elems_per_thread>;
      __shared__ typename BlockMergeSort::TempStorage temp_storage_shuffle;
      BlockMergeSort(temp_storage_shuffle)
          .Sort(thread_data, BeamSortStateComparer());
      __syncthreads();

      // Save from registers to global mem
      for (size_t src = block_offset + threadIdx.x * elems_per_thread, dst = 0;
           dst < elems_per_thread; src++, dst++) {
        if (src < size_) {
          sort_state_[src] = thread_data[dst];
        }
      }
      __syncthreads();
    }

    if (remaining_blocks > 1) {
      cg::sync(grid);
      // 2. copy top half of later blocks into bottom half of earlier blocks
      for (size_t block_id = blockIdx.x; block_id < remaining_blocks;
           block_id += gridDim.x) {
        size_t dst_block_id = remaining_blocks - 1 - block_id;
        if (dst_block_id >= block_id) {
          continue;
        }

        size_t block_offset = block_id * elems_per_block;
        size_t dst_block_offset =
            dst_block_id * elems_per_block + elems_per_block / 2;
        for (size_t block_pos = threadIdx.x; block_pos < elems_per_block / 2;
             block_pos += blockDim.x) {
          size_t src = block_offset + block_pos;
          size_t dst = dst_block_offset + block_pos;
          assert(dst < src);
          if (src < size_) {
            sort_state_[dst] = sort_state_[src];
          }
        }
      }
    } else {
      // 3: done sorting, collect top-k (k=capacity) values at cand's front
      if (blockIdx.x == 0) {
        for (size_t pos = threadIdx.x; pos < min((size_t)size_, capacity_);
             pos += blockDim.x) {
          auto unsorted_pos = sort_state_[pos].pos_;
          topk_cand_[pos] = cand_[unsorted_pos];
        }
        __syncthreads();
        for (size_t pos = threadIdx.x; pos < min((size_t)size_, capacity_);
             pos += blockDim.x) {
          cand_[pos] = topk_cand_[pos];
        }
      }
    }
    cg::sync(grid);

    // half number of blocks and loop around
    remaining_blocks =
        remaining_blocks == 1 ? 0 : (remaining_blocks - 1) / 2 + 1;
  }
}

__device__ void CandidatesGPU::save(
    BeamState* beams, int* next_beam_id, size_t max_beam_id) {
  if (blockIdx.x == 0) {
    __syncthreads();  // to sync with size_ update in trim()
    auto base_beam_id = *next_beam_id;
    for (size_t pos = threadIdx.x; pos < size_; pos += blockDim.x) {
      size_t beam_id = base_beam_id + pos;
      cand_[pos].beam_id_ = beam_id;
      beams[beam_id] = cand_[pos];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      assert(base_beam_id + size_ < max_beam_id);
      atomicAdd(next_beam_id, (int)size_);
    }
  }
}

}  // namespace trie
}  // namespace grimoire
}  // namespace marisa
