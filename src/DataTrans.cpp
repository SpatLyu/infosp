#include <vector>
#include <Rcpp.h>

// Function to convert Rcpp::List to std::vector<std::vector<size_t>> (the `nb` object)
std::vector<std::vector<size_t>> nb2std(const Rcpp::List& nb) {
  // Get the number of elements in the nb object
  size_t n = static_cast<size_t>(nb.size());

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
