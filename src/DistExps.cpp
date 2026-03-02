#include <vector>
#include <cmath>
#include <string>
#include <limits>
#include <numeric>
#include <algorithm>
#include "dist.hpp"
#include <Rcpp.h>

// Wrapper function to compute row-wise distance matrix for an input matrix
// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector RcppDist4Mat(
    const Rcpp::NumericMatrix& mat,
    const std::string& method = "euclidean",
    bool na_rm = true
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

    // Call the distance function
    std::vector<std::vector<double>> distm = Dist::Dist(cppMat, method, na_rm);

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
