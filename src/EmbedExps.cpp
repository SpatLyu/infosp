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

// Wrapper function to calculate lagged values for spatial lattice data
// [[Rcpp::export(rng = false)]]
Rcpp::List RcppLaggedValues4Lattice(const Rcpp::NumericVector& vec,
                                    const Rcpp::List& nb, int lag = 1) {
  int n = nb.size();

  // Convert Rcpp::NumericVector to std::vector<double>
  std::vector<double> vec_std = Rcpp::as<std::vector<double>>(vec);

  // Convert Rcpp::List to std::vector<std::vector<size_t>>
  std::vector<std::vector<size_t>> nb_vec = nb2std(nb);

  // Calculate lagged values
  std::vector<std::vector<double>> lagged_values =
    Embed::LaggedValues4Lattice(vec_std, nb_vec, static_cast<size_t>(std::abs(lag)));

  // Convert std::vector<std::vector<double>> to Rcpp::List
  Rcpp::List result(n);
  for (int i = 0; i < n; ++i) {
    result[i] = Rcpp::wrap(lagged_values[i]);
  }

  return result;
}

// Wrapper function to generate embedding for spatial lattice data
// [[Rcpp::export(rng = false)]]
Rcpp::NumericMatrix RcppGenLatticeEmbedding(const Rcpp::NumericVector& vec,
                                            const Rcpp::List& nb,
                                            int E = 3,
                                            int tau = 1,
                                            int style = 1) {
  // Convert Rcpp::NumericVector to std::vector<double>
  std::vector<double> vec_std = Rcpp::as<std::vector<double>>(vec);

  // Convert Rcpp::List to std::vector<std::vector<size_t>>
  std::vector<std::vector<size_t>> nb_vec = nb2std(nb);

  // Generate embedding
  std::vector<std::vector<double>> embeddings =
    Embed::GenLatticeEmbedding(vec_std, nb_vec,
                               static_cast<size_t>(std::abs(E)),
                               static_cast<size_t>(std::abs(tau)),
                               static_cast<size_t>(std::abs(style)));

  // Convert std::vector<std::vector<double>> to Rcpp::NumericMatrix
  size_t rows = embeddings.size();
  size_t cols = embeddings[0].size();
  Rcpp::NumericMatrix result(rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      result(i, j) = embeddings[i][j];
    }
  }

  return result;
}

// Wrapper function to calculate lagged values for spatial grid data
// [[Rcpp::export(rng = false)]]
Rcpp::NumericMatrix RcppLaggedValues4Grid(const Rcpp::NumericMatrix& mat,
                                          int lag = 1) {
  // Convert Rcpp::NumericMatrix to std::vector<std::vector<double>>
  int numRows = mat.nrow();
  int numCols = mat.ncol();
  std::vector<std::vector<double>> cppMat(numRows, std::vector<double>(numCols));

  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      cppMat[r][c] = mat(r, c);
    }
  }

  // Call the Embed::LaggedValues4Grid function
  std::vector<std::vector<double>> laggedMat =
    Embed::LaggedValues4Grid(cppMat, static_cast<size_t>(std::abs(lag)));

  // Convert the result back to Rcpp::NumericMatrix
  int laggedRows = laggedMat.size();
  int laggedCols = laggedMat[0].size();
  Rcpp::NumericMatrix result(laggedRows, laggedCols);

  for (int r = 0; r < laggedRows; ++r) {
    for (int c = 0; c < laggedCols; ++c) {
      result(r, c) = laggedMat[r][c];
    }
  }

  return result;
}

// Wrapper function to generate embedding for spatial grid data
// [[Rcpp::export(rng = false)]]
Rcpp::NumericMatrix RcppGenGridEmbedding(const Rcpp::NumericMatrix& mat,
                                         int E = 3, int tau = 1, int style = 1) {
  // Convert Rcpp::NumericMatrix to std::vector<std::vector<double>>
  int numRows = mat.nrow();
  int numCols = mat.ncol();
  std::vector<std::vector<double>> cppMat(numRows, std::vector<double>(numCols));

  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      cppMat[r][c] = mat(r, c);
    }
  }

  // Call the GenGridEmbedding function
  std::vector<std::vector<double>> embeddings =
    Embed::GenGridEmbedding(cppMat,
                            static_cast<size_t>(std::abs(E)),
                            static_cast<size_t>(std::abs(tau)),
                            static_cast<size_t>(std::abs(style)));

  // Convert std::vector<std::vector<double>> to Rcpp::NumericMatrix
  int rows = embeddings.size();
  int cols = embeddings[0].size();
  Rcpp::NumericMatrix result(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result(i, j) = embeddings[i][j];
    }
  }

  return result;
}
