#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>
#include <algorithm>
#include "embed.hpp"
#include "symdync.hpp"
#include "infotheo.hpp"
#include "surd.hpp"

double RcppSPE4Lattice(
    const Rcpp::NumericMatrix& mat,
    const Rcpp::IntegerVector& vars,
    const Rcpp::List& nb,
    int E = 3,
    int tau = 1,
    int style = 1,
    bool relative = true,
    double base = 2.0,
    bool na_rm = true
) { 
    const size_t n_cols = static_cast<size_t>(std::abs(mat.ncol()));
    const size_t n_obs = static_cast<size_t>(std::abs(mat.nrow()));

    std::vector<size_t> v = Rcpp::as<std::vector<size_t>>(vars);
    for (auto& idx : v) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Column index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }
    const size_t n_vars = v.size();

    // Convert Rcpp::List to std::vector<std::vector<size_t>>
    std::vector<std::vector<size_t>> nb_std = nb2std(nb);

    std::vector<std::vector<std::vector<uint8_t>>> pm;
    pm.resize(n_vars);
        for (size_t j = 0; j < n_vars; ++j)
            mat[j].reserve(n_obs);

    for (size_t idx : v) {
        std::vector<double> vec(n_obs);
        for (size_t r = 0; r < n_obs; ++r) {
            vec[r] = mat(r, idx);
        }

        // Generate embedding
        std::vector<std::vector<double>> embeddings =
            Embed::GenLatticeEmbedding(vec, nb_std,
                                       static_cast<size_t>(std::abs(E)),
                                       static_cast<size_t>(std::abs(tau)),
                                       static_cast<size_t>(std::abs(style)));
    }


}