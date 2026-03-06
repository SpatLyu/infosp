#include <vector>
#include <cmath>
#include <string>
#include <limits>
#include <numeric>
#include <algorithm>
#include "neighbour.hpp"
#include "DataTrans.h"

// Wrapper function to compute the nearest neighbours for an input feature matrix
// [[Rcpp::export(rng = false)]]
Rcpp::List RcppNN4Mat(
    const Rcpp::NumericMatrix& mat,
    int k,
    std::string& method = "euclidean",
    bool include_self = false
) {
    // Convert Rcpp::NumericMatrix to std::vector<std::vector<double>>
    int numRows = mat.nrow();
    int numCols = mat.ncol();
    std::vector<std::vector<double>> cppMat(numRows, std::vector<double>(numCols));

    for (int r = 0; r < numRows; ++r) {
        for (int c = 0; c < numCols; ++c) {
            cppMat[r][c] = mat(r, c);
        }
    }

    // Call the neighbpurbood function
    std::vector<std::vector<size_t>> neighbours = NN::NN4Mat(
        cppMat, static_cast<size_t>(std::abs(k)), method, include_self);

    // Return nb object (List in R side)
    return std2nb(neighbours);
}
