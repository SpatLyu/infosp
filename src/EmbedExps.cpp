#include <vector>
#include <cmath>
#include <string>
#include <limits>
#include <iterator>
#include <numeric>
#include <algorithm>
#include "embed.hpp"
#include "DataTrans.h"

// Wrapper function to calculate accumulated lagged neighbor indices for spatial lattice data
// [[Rcpp::export(rng = false)]]
Rcpp::List RcppLaggedNeighbors4Lattice(const Rcpp::List& nb, int lag = 1) {
  // Convert Rcpp::List to std::vector<std::vector<size_t>>
  std::vector<std::vector<size_t>> nb_vec = nb2std(nb);

  // Calculate lagged indices
  std::vector<std::vector<size_t>> lagged_indices =
    Embed::LaggedNeighbors4Lattice(nb_vec, static_cast<size_t>(std::abs(lag)));

  // Return nb object (List in R side)
  return std2nb(lagged_indices);
}
