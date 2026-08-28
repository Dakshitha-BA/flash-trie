#include <marisa.h>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>

#include "beam-search-helper.h"
#include "cmdopt.h"

using namespace marisa;

namespace {

bool mmap_flag = false;
bool test_cpu = true;
bool test_gpu = false;
bool check_accuracy = false;
bool early_exit = false;

void print_help(const char* cmd) {
  std::cerr
      << "Usage: " << cmd
      << " [OPTION]... DATASET_DIR\n\n"
         "Options:\n"
         "  -m, --mmap-dictionary  use memory-mapped I/O to load a dictionary\n"
         "  -r, --read-dictionary  read an entire dictionary into memory"
         " (default)\n"
         "  -g, --test-gpu         test gpu beam search\n"
         "  -a, --check-accuracy   compare cpu and gpu beam search output\n"
         "  -e, --early-exit       early exit during beam search\n"
         "  -h, --help             print this help\n"
      << std::endl;
}

void print_result(std::size_t req_id,
    const std::vector<std::vector<Label>>& out_sent,
    const std::vector<double>& out_logp) {
  return;
  for (std::size_t beam = 0; beam < out_sent.size(); beam++) {
    std::cout << out_logp[beam];
    for (auto token : out_sent[beam]) {
      std::cout << " " << token;
    }
    std::cout << std::endl;
  }
}

void filter_invalid_beams(const Trie& trie,
    std::vector<std::vector<Label>>& out_sent, std::vector<double>& out_logp,
    std::vector<size_t>& ids, bool is_cpu) {
  assert(ids.empty());

  std::vector<std::vector<Label>> valid_out_sent;
  std::vector<double> valid_out_logp;

  for (std::size_t beam = 0; beam < out_sent.size(); beam++) {
    Agent agent;
    agent.set_query(out_sent[beam].data(), out_sent[beam].size());
    if (trie.lookup(agent)) {
      valid_out_sent.push_back(out_sent[beam]);
      valid_out_logp.push_back(out_logp[beam]);
      ids.push_back(agent.key().id());
    }
  }
  std::cout << (is_cpu ? "CPU" : "GPU") << " valid beams "
            << valid_out_sent.size() << "/" << out_sent.size() << std::endl;

  if (not is_cpu and (valid_out_sent.size() != out_sent.size())) {
    std::cout << "GPU beam is invalid" << std::endl;
    assert(false);
  }

  out_sent = std::move(valid_out_sent);
  out_logp = std::move(valid_out_logp);
}

size_t find_cpu_beams_in_gpu_beams(const std::vector<size_t>& cpu_ids,
    const std::vector<size_t>& gpu_ids, const std::vector<double>& cpu_logp,
    const std::vector<double>& gpu_logp) {
  assert(cpu_ids.size() <= gpu_ids.size());
  size_t num_matches = 0;
  double atol = 1e-4;
  for (size_t cpu_beam = 0; cpu_beam < cpu_ids.size(); cpu_beam++) {
    for (size_t gpu_beam = 0; gpu_beam < gpu_ids.size(); gpu_beam++) {
      if (cpu_ids[cpu_beam] == gpu_ids[gpu_beam]) {
        if (std::fabs(cpu_logp[cpu_beam] - gpu_logp[gpu_beam]) < atol) {
          num_matches++;
          break;
        }
      }
    }
  }
  return num_matches;
}

bool compare_result(const Trie& trie, std::size_t req_id,
    std::vector<std::vector<Label>>& cpu_out_sent,
    std::vector<double>& cpu_out_logp,
    std::vector<std::vector<Label>>& gpu_out_sent,
    std::vector<double>& gpu_out_logp) {
  std::cout << "Checking accuracy for req_id " << req_id << std::endl;

  std::vector<size_t> cpu_ids, gpu_ids;
  filter_invalid_beams(trie, cpu_out_sent, cpu_out_logp, cpu_ids, true);
  filter_invalid_beams(trie, gpu_out_sent, gpu_out_logp, gpu_ids, false);

  if (cpu_out_sent.size() != gpu_out_sent.size()) {
    printf("Mismatch in valid beam count, cpu %lu != gpu %lu\n",
        cpu_out_sent.size(), gpu_out_sent.size());
    fflush(stdout);
  }
  auto num_matches =
      find_cpu_beams_in_gpu_beams(cpu_ids, gpu_ids, cpu_out_logp, gpu_out_logp);
  if (num_matches < cpu_ids.size()) {
    printf("%lu/%lu CPU beams not found in GPU beams\n",
        cpu_ids.size() - num_matches, cpu_ids.size());
    fflush(stdout);
  }
  return true;

  for (std::size_t beam = 0; beam < cpu_out_sent.size(); beam++) {
    if (cpu_out_logp[beam] != gpu_out_logp[beam]) {
      printf("Mismatch in beam %lu score cpu %f != gpu %f\n", beam,
          cpu_out_logp[beam], gpu_out_logp[beam]);
      return false;
    }
    if (cpu_out_sent[beam].size() != gpu_out_sent[beam].size()) {
      printf("Mismatch in beam %lu length cpu %lu != gpu %lu\n", beam,
          cpu_out_sent[beam].size(), gpu_out_sent[beam].size());
      return false;
    }
    for (std::size_t token_id = 0; token_id < cpu_out_sent[beam].size();
         token_id++) {
      auto cpu_token = cpu_out_sent[beam][token_id];
      auto gpu_token = gpu_out_sent[beam][token_id];
      if (cpu_token != gpu_token) {
        printf("Mismatch in beam %lu token_id %lu cpu %u != gpu %u\n", beam,
            token_id, cpu_token, gpu_token);
        return false;
      }
    }
  }
  return true;
}

int beam_search(const char* const* args, std::size_t num_args) {
  if (num_args == 0) {
    std::cerr << "error: dictionary is not specified" << std::endl;
    return 10;
  } else if (num_args > 3) {
    std::cerr << "more than 3 arguments are not allowed" << std::endl;
    return 11;
  }

  std::string dataset_dir(args[0]);
  std::string trie_path = dataset_dir + "/marisa_trie.bin";
  std::cout << "Trie path " << trie_path << std::endl;
  Trie trie;
  if (mmap_flag) {
    try {
      trie.mmap(trie_path.c_str());
    } catch (const Exception& ex) {
      std::cerr << ex.what()
                << ": failed to mmap a dictionary file: " << args[0]
                << std::endl;
      return 20;
    }
  } else {
    try {
      trie.load(trie_path.c_str());
    } catch (const Exception& ex) {
      std::cerr << ex.what()
                << ": failed to load a dictionary file: " << args[0]
                << std::endl;
      return 21;
    }
  }
  std::cout << "Finished loading trie" << std::endl;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  std::cout << "GPU SMs " << deviceProp.multiProcessorCount << std::endl;

  auto keys_and_scores = load_topk_keys_and_scores(dataset_dir);
  auto topk_ids = std::get<0>(keys_and_scores);
  auto topk_logps = std::get<1>(keys_and_scores);

  auto num_requests = topk_ids.size();
  auto max_output_len = topk_ids[0].size();

  float token_logp_threshold = -7.0;
  double sent_logp_threshold = -40.0;
  double length_norm = 5;
  auto beam_width = num_args >= 2 ? std::stoi(args[1]) : 200;
  std::vector<UInt32> beams(max_output_len, beam_width);

  size_t batch_size = num_args >= 3 ? std::stoi(args[2]) : 1;
  if (test_gpu) {
    trie.init_tbs_gpu(batch_size);
  }
  std::size_t test_repeats = 1;
  std::size_t num_test_requests = 100;
  std::size_t first_req_id = 0;
  std::size_t last_req_id =
      std::min(num_test_requests + first_req_id, num_requests);
  std::vector<double> times;

  for (std::size_t repeat = 0; repeat < test_repeats; repeat++) {
    for (std::size_t req_id = first_req_id; req_id < last_req_id; req_id++) {
      std::vector<std::vector<Label>>& topk_id = topk_ids[req_id];
      std::vector<std::vector<float>>& topk_logp = topk_logps[req_id];

      // CPU run
      std::vector<std::vector<Label>> cpu_out_sent;
      std::vector<double> cpu_out_logp_norm;

      if (test_cpu) {
        auto start = std::chrono::high_resolution_clock::now();
        TbsInput input{beams, topk_id, topk_logp, cpu_out_sent,
            cpu_out_logp_norm, token_logp_threshold, sent_logp_threshold,
            length_norm, early_exit};
        trie.tbs(input);
        auto end = std::chrono::high_resolution_clock::now();

        if (test_repeats == 1 or repeat > 0) {
          auto time = get_msec(start, end);
          if (not check_accuracy) {
            printf("Req #%lu: %lu beams generated in %.2f msec\n", req_id,
                cpu_out_sent.size(), time);
          }
          times.push_back(time);
          print_result(req_id, cpu_out_sent, cpu_out_logp_norm);
        }
      }

      // GPU run
      std::vector<std::vector<Label>> gpu_out_sent;
      std::vector<double> gpu_out_logp_norm;

      if (test_gpu) {
        TbsInput input{beams, topk_id, topk_logp, gpu_out_sent,
            gpu_out_logp_norm, token_logp_threshold, sent_logp_threshold,
            length_norm, early_exit};

        trie.tbs_gpu(input);
        print_result(req_id, gpu_out_sent, gpu_out_logp_norm);
      }

      if (check_accuracy) {
        bool match = compare_result(trie, req_id, cpu_out_sent,
            cpu_out_logp_norm, gpu_out_sent, gpu_out_logp_norm);
        if (!match) {
          printf("Mismatch in req_id %lu\n", req_id);
          fflush(stdout);
        }
      }
    }
  }

  if (test_cpu) {
    auto req_time =
        std::accumulate(times.begin(), times.end(), 0.0) / (double)times.size();
    printf("Measured time per request in CPU tbs: %.2f msec\n", req_time);
  }
  if (!test_gpu) {
    return 0;
  }

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<std::vector<Label>>> batch_out_sent(batch_size);
  std::vector<std::vector<double>> batch_out_logp_norm(batch_size);

  // Optional per-request timing-breakdown output. Set env TBS_TIMING_CSV to a
  // path to emit one row per request with the GPU phase breakdown (populated
  // only when the library is built with -DTBS_PROFILE).
  const char* timing_csv_path = std::getenv("TBS_TIMING_CSV");
  std::ofstream timing_csv;
  if (timing_csv_path != nullptr) {
    timing_csv.open(timing_csv_path);
    timing_csv << "req_id,beam_width,batch_size,num_beams,kernel_ms,"
                  "host_batch_ms,t_expansion_ms,t_validation_ms,"
                  "t_selection_ms,t_grid_overhead_ms\n";
    std::cout << "Writing per-request timing breakdown to " << timing_csv_path
              << std::endl;
  }

  size_t num_batches = 0;
  for (std::size_t req_id = first_req_id; req_id < last_req_id;
       req_id += batch_size, num_batches++) {
    std::vector<TbsInput> inputs;

    printf("Reqs:");
    size_t cur_batch_size = std::min(last_req_id - req_id, batch_size);
    for (size_t batch_id = 0; batch_id < cur_batch_size; batch_id++) {
      auto cur_req_id = req_id + batch_id;
      printf(" %lu", cur_req_id);
      inputs.push_back(TbsInput{beams, topk_ids[cur_req_id],
          topk_logps[cur_req_id], batch_out_sent[batch_id],
          batch_out_logp_norm[batch_id], token_logp_threshold,
          sent_logp_threshold, length_norm, early_exit});
    }
    printf(" ");

    auto batch_start = std::chrono::high_resolution_clock::now();
    trie.tbs_gpu_batched(inputs);
    auto batch_end = std::chrono::high_resolution_clock::now();
    auto batch_msec = get_msec(batch_start, batch_end);

    printf("Beams:");
    for (size_t batch_id = 0; batch_id < cur_batch_size; batch_id++) {
      printf(" %lu", batch_out_sent[batch_id].size());
    }
    printf(" Time: %.2f msec\n", batch_msec);

    if (timing_csv.is_open()) {
      for (size_t batch_id = 0; batch_id < cur_batch_size; batch_id++) {
        const auto& in = inputs[batch_id];
        timing_csv << (req_id + batch_id) << ',' << beam_width << ','
                   << cur_batch_size << ',' << batch_out_sent[batch_id].size()
                   << ',' << in.kernel_ms << ',' << batch_msec << ','
                   << in.t_expansion_ms << ',' << in.t_validation_ms << ','
                   << in.t_selection_ms << ',' << in.t_grid_overhead_ms << '\n';
      }
    }
  }
  if (timing_csv.is_open()) {
    timing_csv.close();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto loop_time = get_msec(start, end);
  printf("Total time to process %lu requests with %lu batch size: %.2f msec\n",
      last_req_id - first_req_id, batch_size, loop_time);
  printf("Per batch latency  : %.2f msec\n", loop_time / (double)num_batches);
  printf("Per request latency: %.2f msec\n",
      loop_time / (double)(last_req_id - first_req_id));

  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  std::ios::sync_with_stdio(false);

  ::cmdopt_option long_options[] = {{"mmap-dictionary", 0, NULL, 'm'},
      {"read-dictionary", 0, NULL, 'r'}, {"test-gpu", 0, NULL, 'g'},
      {"check-accuracy", 0, NULL, 'a'}, {"help", 0, NULL, 'h'},
      {NULL, 0, NULL, 0}};
  ::cmdopt_t cmdopt;
  ::cmdopt_init(&cmdopt, argc, argv, "n:mrgaeh", long_options);
  int label;
  while ((label = ::cmdopt_get(&cmdopt)) != -1) {
    switch (label) {
      case 'm': {
        mmap_flag = true;
        break;
      }
      case 'r': {
        mmap_flag = false;
        break;
      }
      case 'g': {
        test_cpu = false;
        test_gpu = true;
        break;
      }
      case 'a': {
        check_accuracy = true;
        test_cpu = test_gpu = true;
        break;
      }
      case 'e': {
        early_exit = true;
        break;
      }
      case 'h': {
        print_help(argv[0]);
        return 0;
      }
      default: {
        return 1;
      }
    }
  }
  return beam_search(cmdopt.argv + cmdopt.optind,
      static_cast<std::size_t>(cmdopt.argc - cmdopt.optind));
}
