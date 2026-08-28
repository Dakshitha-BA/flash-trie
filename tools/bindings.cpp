#include <marisa.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "beam-search-helper.h"

namespace py = pybind11;
using namespace marisa;

void build_from_list(
    Trie &trie, const std::vector<std::vector<uint32_t>> &token_rows) {
  Keyset keyset;

  for (const auto &row : token_rows) {
    keyset.push_back(row.data(), row.size());
  }

  int num_tries = 1;
  int config_flags = (num_tries | MARISA_DEFAULT_TAIL | MARISA_LABEL_ORDER |
                      MARISA_DEFAULT_CACHE);
  trie.build(keyset, config_flags);
}

PYBIND11_MODULE(flashtrie, m) {
  typedef marisa::Trie Trie;

  py::class_<Trie>(m, "Trie")
      .def(py::init<>())
      .def("load", &Trie::load)
      .def("mmap", &Trie::mmap)
      .def("build", &build_from_list, py::arg("token_rows"))
      .def("total_size", &Trie::total_size)
      .def("num_tries", &Trie::num_tries)
      .def("num_keys", &Trie::num_keys)
      .def("num_nodes", &Trie::num_nodes)
      .def("save", &Trie::save)
      .def("clear", &Trie::clear)
      .def("init_tbs_gpu", &Trie::init_tbs_gpu, py::arg("max_batch_size"),
          py::arg("max_num_topk") = 1024, py::arg("max_num_outpos") = 64,
          py::arg("max_num_out_beams") = 8192,
          py::arg("max_num_out_tokens") = 8192 * 64,
          py::arg("max_num_inter_beams") = 1024 * 1024,
          py::arg("radix_topk_threshold") = 1024);

  m.def(
      "tbs",
      [](Trie &trie, const std::vector<uint32_t> &beams,
          const std::vector<std::vector<Label>> &topk_id,
          const std::vector<std::vector<float>> &topk_logp,
          float token_logp_threshold, double sent_logp_threshold,
          double length_norm, bool early_exit, bool use_gpu) {
        std::vector<std::vector<Label>> out_sent;
        std::vector<double> out_logp_norm;
        auto start = std::chrono::high_resolution_clock::now();
        TbsInput input{beams, topk_id, topk_logp, out_sent, out_logp_norm,
            token_logp_threshold, sent_logp_threshold, length_norm, early_exit};

        if (use_gpu) {
          trie.tbs_gpu(input);
        } else {
          trie.tbs(input);
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::make_tuple(out_sent, out_logp_norm, get_msec(start, end));
      },
      py::arg("trie"), py::arg("beams"), py::arg("topk_id"),
      py::arg("topk_logp"), py::arg("token_logp_threshold"),
      py::arg("sents_logp_threshold"), py::arg("length_norm"),
      py::arg("early_exit"), py::arg("use_gpu") = false);

  m.def(
      "bs_tbs_gpu",
      [](Trie &trie, size_t batch_size,
          const std::vector<std::vector<uint32_t>> &batch_beams,
          const std::vector<std::vector<std::vector<Label>>> &batch_topk_id,
          const std::vector<std::vector<std::vector<float>>> &batch_topk_logp,
          float token_logp_threshold, double sent_logp_threshold,
          double length_norm, bool early_exit) {
        assert(batch_size <= trie.get_max_batch_size());
        assert(batch_size == batch_topk_id.size());
        assert(batch_size == batch_topk_logp.size());

        std::vector<TbsInput> inputs;
        std::vector<std::vector<std::vector<Label>>> batch_out_sent;
        std::vector<std::vector<double>> batch_out_logp_norm;

        batch_out_sent.resize(batch_size);
        batch_out_logp_norm.resize(batch_size);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < batch_size; ++i) {
          TbsInput input{batch_beams[i], batch_topk_id[i], batch_topk_logp[i],
              batch_out_sent[i], batch_out_logp_norm[i], token_logp_threshold,
              sent_logp_threshold, length_norm, early_exit};
          inputs.push_back(std::move(input));
        }

        trie.tbs_gpu_batched(inputs);

        auto end = std::chrono::high_resolution_clock::now();

        // Per-request GPU phase breakdown (ms): one row per request,
        // columns = [expansion, validation, selection, grid/overhead, kernel].
        // Values are nonzero only when built with -DTBS_PROFILE.
        std::vector<std::vector<double>> phase_times;
        phase_times.reserve(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
          phase_times.push_back({inputs[i].t_expansion_ms,
              inputs[i].t_validation_ms, inputs[i].t_selection_ms,
              inputs[i].t_grid_overhead_ms, inputs[i].kernel_ms});
        }

        return std::make_tuple(batch_out_sent, batch_out_logp_norm,
            get_msec(start, end), phase_times);
      },
      py::arg("trie"), py::arg("batch_size"), py::arg("batch_beams"),
      py::arg("batch_topk_id"), py::arg("batch_topk_logp"),
      py::arg("token_logp_threshold"), py::arg("sents_logp_threshold"),
      py::arg("length_norm"), py::arg("early_exit"));

  m.def("load_topk_keys_and_scores", &load_topk_keys_and_scores);
}
