#include <vector>
#include <cmath>
#include <string>
#include <limits>
#include <numeric>
#include <algorithm>
#include "distance.hpp"
#include <Rcpp.h>

// Wrapper function to compute the nearest neighbours for an input feature matrix
// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector RcppDist4Mat(
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
    std::vector<std::vector<size_t>> distm = NN::NN4Mat(
        cppMat, static_cast<size_t>(std::abs(k)), method, include_self);

    // Convert std::vector<std::vector<double>> to Rcpp::NumericMatrix
    int rows = distm.size();
    int cols = distm[0].size();
    Rcpp::NumericMatrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
        result(i, j) = distm[i][j];
        }
    }

    return result;
}