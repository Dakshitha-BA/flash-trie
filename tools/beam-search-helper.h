#include <algorithm>
#include <fstream>
#include <numeric>

using namespace marisa;

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(
    const std::vector<T>& vec, Compare compare) {
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(),
      [&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j]); });
  return p;
}

template <typename T>
std::vector<T> apply_permutation(
    const std::vector<T>& vec, const std::vector<std::size_t>& p) {
  std::vector<T> sorted_vec(vec.size());
  std::transform(p.begin(), p.end(), sorted_vec.begin(),
      [&](std::size_t i) { return vec[i]; });
  return sorted_vec;
}

std::tuple<std::vector<std::vector<std::vector<Label>>>,
    std::vector<std::vector<std::vector<float>>>>
load_topk_keys_and_scores(std::string dataset_dir) {
  std::string keys_path = dataset_dir + "/topk_keys.txt";
  std::string scores_path = dataset_dir + "/topk_scores.txt";
  std::cout << "Loading topk keys from " << keys_path << " and scores from "
            << scores_path << std::endl;

  std::ifstream keys_file(keys_path);
  std::ifstream scores_file(scores_path);

  if (!keys_file.is_open()) {
    printf("Error opening topk keys\n");
    exit(1);
  }
  if (!scores_file.is_open()) {
    printf("Error opening topk scores\n");
    exit(1);
  }

  std::string keys_line, scores_line, buf;
  getline(keys_file, keys_line);
  std::stringstream keys_metadata_ss(keys_line);

  getline(scores_file, scores_line);
  std::stringstream scores_metadata_ss(scores_line);

  uint32_t metadata[3];
  for (uint32_t i = 0; i < 3; i++) {
    keys_metadata_ss >> buf;
    metadata[i] = stoi(buf);
    scores_metadata_ss >> buf;
    assert((uint32_t)stoi(buf) == metadata[i]);
  }
  const uint32_t num_requests = metadata[0];
  const uint32_t max_output_len = metadata[1];
  const uint32_t topk = metadata[2];
  std::cout << "Parsed num_requests " << num_requests << " max_output_len "
            << max_output_len << " topk " << topk << std::endl;

  const uint32_t final_topk = topk;
  if (final_topk < topk) {
    std::cout << "Truncating topk from " << topk << " in input to "
              << final_topk << std::endl;

  } else if (final_topk > topk) {
    std::cout << "Extending topk from " << topk << " in input to " << final_topk
              << std::endl;
  }

  std::vector<std::vector<std::vector<Label>>> keys(num_requests);
  std::vector<std::vector<std::vector<float>>> scores(num_requests);

  for (uint32_t topk_id = 0; topk_id < num_requests; topk_id++) {
    keys[topk_id].resize(max_output_len);
    scores[topk_id].resize(max_output_len);

    for (uint32_t depth = 0; depth < max_output_len; depth++) {
      auto& level_keys = keys[topk_id][depth];
      getline(keys_file, keys_line);
      std::stringstream keys_ss(keys_line);
      while (keys_ss >> buf) {
        level_keys.push_back(static_cast<Label>(stof(buf)));
      }
      assert(level_keys.size() == topk);

      auto& level_scores = scores[topk_id][depth];
      getline(scores_file, scores_line);
      std::stringstream scores_ss(scores_line);
      while (scores_ss >> buf) {
        level_scores.push_back(stof(buf));
      }
      assert(level_scores.size() == topk);

      if (depth > 0) {
        assert(is_sorted(
            level_scores.begin(), level_scores.end(), std::greater<float>()));
        auto perm = sort_permutation(level_keys, std::less<Label>());
        level_keys = apply_permutation(level_keys, perm);
        level_scores = apply_permutation(level_scores, perm);
      }

      if (final_topk < level_keys.size()) {  // Truncate
        level_keys.resize(final_topk);
        level_scores.resize(final_topk);
      } else {  // Extend
        while (level_keys.size() < final_topk) {
          level_keys.push_back(std::numeric_limits<uint32_t>::max());
          level_scores.push_back(-std::numeric_limits<float>::max());
        }
      }
    }
  }
  return std::make_tuple(keys, scores);
}
