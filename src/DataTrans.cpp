#include <vector>
#include <Rcpp.h>

// Function to convert Rcpp::List to std::vector<std::vector<size_t>> (the `nb` object)
std::vector<std::vector<size_t>> nb2std(const Rcpp::List& nb) {
  // Get the number of elements in the nb object
  size_t n = static_cast<size_t>(nb.size());
  if (n <= 1) {
    Rcpp::stop("The nb object must contain at least two spatial units (got %d)", n);
  }
  
  // Create a std::vector<std::vector<size_t>> to store the result
  std::vector<std::vector<size_t>> result(n);

  // Iterate over each element in the nb object
  for (size_t i = 0; i < n; ++i) {
    // Get the current element (should be an integer vector)
    Rcpp::IntegerVector current_nb = nb[i];
    size_t cur_num_nb = static_cast<size_t>(current_nb.size());

    // Create a vector<size_t> to store the current subset of elements
    std::vector<size_t> current_subset;
    current_subset.reserve(cur_num_nb);

    // Iterate over each element in the current subset
    for (size_t j = 0; j < cur_num_nb; ++j) {
      // Subtract one from each element to convert from R's 1-based indexing to C++'s 0-based indexing
      current_subset.push_back(current_nb[j] - 1);
    }

    // Add the current subset to the result
    result[i] = current_subset;
  }

  return result;
}

// Function to convert std::vector<std::vector<size_t>> (the `nb` object) to Rcpp::List
Rcpp::List std2nb(const std::vector<std::vector<size_t>>& nb) {
  size_t n = nb.size();
  Rcpp::List result(n);

  for (size_t i = 0; i < n; ++i) {
    const auto& neighbors = nb[i];
    Rcpp::IntegerVector r_neighbors(neighbors.size());
    for (size_t j = 0; j < neighbors.size(); ++j) {
      r_neighbors[j] = static_cast<int>(neighbors[j] + 1);
    }
    result[i] = r_neighbors;
  }

  return result;
}
