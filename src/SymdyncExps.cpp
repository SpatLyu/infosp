#include <vector>
#include <cmath>
#include <string>
#include <limits>
#include <iterator>
#include <numeric>
#include <algorithm>
#include "symdync.hpp"
#include "DataTrans.h"

// Wrapper function to convert a continuous embedding matrix into signature matrix
// [[Rcpp::export(rng = false)]]
Rcpp::NumericMatrix RcppGenSignatureSpace(
    const Rcpp::NumericMatrix& mat,
    bool relative = true
)
{
    const size_t n_rows = mat.nrow();
    const size_t n_cols = mat.ncol();

    if (n_cols < 2)
        Rcpp::stop("Matrix must have at least 2 columns.");

    std::vector<std::vector<double>> input(
        n_rows,
        std::vector<double>(n_cols)
    );

    for (size_t i = 0; i < n_rows; ++i)
        for (size_t j = 0; j < n_cols; ++j)
            input[i][j] = mat(i, j);

    auto result = SymDync::GenSignatureSpace(input, relative);

    const size_t out_cols = result[0].size();
    Rcpp::NumericMatrix out(n_rows, out_cols);

    for (size_t i = 0; i < n_rows; ++i)
        for (size_t j = 0; j < out_cols; ++j)
            out(i, j) = result[i][j];

    return out;
}

// Wrapper function to convert a continuous signature matrix into symbolic pattern matrix
// [[Rcpp::export(rng = false)]]
Rcpp::CharacterMatrix RcppGenPatternSpace(
    Rcpp::NumericMatrix mat,
    bool NA_rm = true
)
{
    const size_t n_rows = mat.nrow();
    const size_t n_cols = mat.ncol();

    std::vector<std::vector<double>> input(
        n_rows,
        std::vector<double>(n_cols)
    );

    for (size_t i = 0; i < n_rows; ++i)
        for (size_t j = 0; j < n_cols; ++j)
            input[i][j] = mat(i, j);

    auto pat = SymDync::GenPatternSpace(input, NA_rm);
    return pat2vec(pat);
}