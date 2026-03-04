/**
 * @brief Computes k-nearest neighbor indices for prediction rows from a given library.
 *
 * For each index in `pred`, this function finds up to `k` nearest neighbors
 * from the index set `lib`, based on a precomputed distance matrix.
 *
 * If `include_self` is true and the prediction index is also contained in `lib`,
 * the index itself will be placed as the first neighbor. Otherwise, self will
 * be excluded from the candidate neighbors.
 *
 * The output vector has size equal to distmat.size(). For indices not in `pred`,
 * the corresponding entry is an empty vector.
 *
 * @param distmat       Precomputed n x n distance matrix (may contain NaN).
 * @param pred          Indices for which neighbors should be computed.
 * @param lib           Candidate neighbor indices.
 * @param k             Number of neighbors to retain.
 * @param include_self  Whether to include the index itself if it is in lib.
 *
 * @return A vector of length n. For each i in pred, contains up to k neighbor
 *         indices from lib sorted by increasing distance. Other rows are empty.
 */
std::vector<std::vector<size_t>> NN4DistMat(
    const std::vector<std::vector<double>>& distmat,
    const std::vector<size_t>& pred,
    const std::vector<size_t>& lib,
    size_t k,
    bool include_self = false)
{
  const size_t n = distmat.size();

  // Initialize result with empty vectors
  std::vector<std::vector<size_t>> result(n);

  // Build lib membership lookup
  std::unordered_set<size_t> lib_set(lib.begin(), lib.end());

  for (size_t i : pred) {

    // if (i >= n || distmat[i].size() != n) continue;

    const auto& row = distmat[i];

    // if (std::isnan(row[i])) continue;

    std::vector<std::pair<double, size_t>> candidates;
    bool self_in_lib = (lib_set.find(i) != lib_set.end());

    // Collect valid neighbors from lib excluding self
    for (size_t j : lib) {

      if (i == j) continue;

      double d = row[j];
      if (!std::isnan(d)) {
        candidates.emplace_back(d, j);
      }
    }

    std::vector<size_t> indices;
    indices.reserve(k);
    size_t effective_k = k;

    // Handle self inclusion
    if (include_self && self_in_lib) {
      indices.push_back(i);
      if (effective_k > 0) {
        effective_k -= 1;
      }
    }

    size_t num_neighbors = std::min(effective_k, candidates.size());

    if (num_neighbors > 0) {

      std::partial_sort(
        candidates.begin(),
        candidates.begin() + num_neighbors,
        candidates.end(),
        [](const std::pair<double, size_t>& a,
           const std::pair<double, size_t>& b) {
          if (!doubleNearlyEqual(a.first, b.first)) {
            return a.first < b.first;
          } else {
            return a.second < b.second;
          }
        }
      );

      for (size_t m = 0; m < num_neighbors; ++m) {
        indices.push_back(candidates[m].second);
      }
    }

    result[i] = std::move(indices);
  }

  return result;
}

std::vector<std::vector<size_t>> NN4DistMat(
    const std::vector<std::vector<double>>& distmat,
    size_t k,
    bool include_self = false)
{
  const size_t n = distmat.size();

  // Initialize result with empty vectors
  std::vector<std::vector<size_t>> result(n);

  for (size_t i = 0; i < n; ++i) {
    const auto& row = distmat[i];

    // if (std::isnan(row[i])) continue;

    std::vector<std::pair<double, size_t>> candidates;

    for (size_t j = 0; j < n; ++j) {

      if (i == j) continue;

      double d = row[j];
      if (!std::isnan(d)) {
        candidates.emplace_back(d, j);
      }
    }

    std::vector<size_t> indices;
    indices.reserve(k);
    size_t effective_k = k;

    // Handle self inclusion
    if (include_self) {
      indices.push_back(i);
      if (effective_k > 0) {
        effective_k -= 1;
      }
    }

    size_t num_neighbors = std::min(effective_k, candidates.size());

    if (num_neighbors > 0) {

      std::partial_sort(
        candidates.begin(),
        candidates.begin() + num_neighbors,
        candidates.end(),
        [](const std::pair<double, size_t>& a,
           const std::pair<double, size_t>& b) {
          if (!doubleNearlyEqual(a.first, b.first)) {
            return a.first < b.first;
          } else {
            return a.second < b.second;
          }
        }
      );

      for (size_t m = 0; m < num_neighbors; ++m) {
        indices.push_back(candidates[m].second);
      }
    }

    result[i] = std::move(indices);
  }

  return result;
}
