#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <unordered_set>
#include "embed.hpp"
#include "symdync.hpp"
#include "infotheo.hpp"
#include "surd.hpp"
#include "DataTrans.h"

// Wrapper function to calculate pattern entropy for spatial lattice data 
// [[Rcpp::export(rng = false)]]
double RcppSPE4Lattice(
    const Rcpp::NumericMatrix& mat,
    const Rcpp::IntegerVector& vars,
    const Rcpp::List& nb,
    const Rcpp::IntegerVector& E,
    const Rcpp::IntegerVector& tau,
    const Rcpp::IntegerVector& style,
    bool relative = true,
    double base = 2.0,
    bool na_rm = true
) {
    // Basic matrix dimensions
    const size_t n_cols = static_cast<size_t>(mat.ncol());
    const size_t n_obs  = static_cast<size_t>(mat.nrow());

    // Convert R variable indices -> C++ (0-based)
    std::vector<size_t> v = Rcpp::as<std::vector<size_t>>(vars);
    for (auto& idx : v)
    {
        if (idx < 1 || idx > n_cols)
        {
            Rcpp::stop(
                "Column index %d out of bounds [1, %d]",
                static_cast<int>(idx),
                static_cast<int>(n_cols)
            );
        }

        idx -= 1;  // convert to 0-based indexing
    }

    const size_t n_vars = v.size();

    if (n_vars == 0 || n_obs == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Convert R neighbor structure -> std::vector<std::vector<size_t>>
    std::vector<std::vector<size_t>> nb_std = nb2std(nb);

    // Pattern matrix
    std::vector<std::vector<std::vector<uint8_t>>> pm;
    pm.resize(n_vars);

    // Process each selected variable
    for (size_t j = 0; j < n_vars; ++j)
    {
        size_t col_id = v[j];

        // Extract column vector from R matrix
        std::vector<double> vec(n_obs);
        for (size_t r = 0; r < n_obs; ++r)
        {
            vec[r] = mat(r, col_id);
        }

        // Generate lattice embedding
        std::vector<std::vector<double>> embeddings =
            Embed::GenLatticeEmbedding(
                vec,
                nb_std,
                static_cast<size_t>(std::abs(E[j])),
                static_cast<size_t>(std::abs(tau[j])),
                static_cast<size_t>(std::abs(style[j]))
            );

        // Convert continuous embedding -> symbolic patterns
        pm[j] = SymDync::GenSymbolicPattern(
            embeddings,
            relative,
            na_rm
        );
    }

    // Construct variable index vector for JE
    //
    // pm layout:
    //   pm[0], pm[1], ..., pm[n_vars-1]
    //
    // Therefore JE variables must be:
    //   {0,1,...,n_vars-1}
    std::vector<size_t> je_vars(n_vars);
    for (size_t i = 0; i < n_vars; ++i)
    {
        je_vars[i] = i;
    }

    // Compute Joint Entropy
    return InfoTheo::JE(pm, je_vars, base, na_rm);
}

// Wrapper function to calculate pattern entropy for spatial grid data 
// [[Rcpp::export(rng = false)]]
double RcppSPE4Grid(
    const Rcpp::NumericMatrix& mat,
    const Rcpp::IntegerVector& vars,
    const Rcpp::IntegerVector& E,
    const Rcpp::IntegerVector& tau,
    const Rcpp::IntegerVector& style,
    int nrows,
    bool relative = true,
    double base = 2.0,
    bool na_rm = true
) {
    // Basic matrix dimensions
    const size_t n_cols = static_cast<size_t>(mat.ncol());
    const size_t n_obs  = static_cast<size_t>(mat.nrow());

    // Convert R variable indices -> C++ (0-based)
    std::vector<size_t> v = Rcpp::as<std::vector<size_t>>(vars);
    for (auto& idx : v)
    {
        if (idx < 1 || idx > n_cols)
        {
            Rcpp::stop(
                "Column index %d out of bounds [1, %d]",
                static_cast<int>(idx),
                static_cast<int>(n_cols)
            );
        }

        idx -= 1;  // convert to 0-based indexing
    }

    const size_t n_vars = v.size();

    if (n_vars == 0 || n_obs == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Pattern matrix
    std::vector<std::vector<std::vector<uint8_t>>> pm;
    pm.resize(n_vars);

    // Process each selected variable
    for (size_t j = 0; j < n_vars; ++j)
    {
        size_t col_id = v[j];

        // Extract subset matrix from R matrix
        std::vector<std::vector<double>> cm(
            nrows, std::vector<double>(n_obs / nrows));
        for (size_t r = 0; r < n_obs; ++r)
        {
            cm[r % nrows][r / nrows] = mat(r, col_id);
        }

        // Generate grid embedding
        std::vector<std::vector<double>> embeddings =
            Embed::GenGridEmbedding(
                cm,
                static_cast<size_t>(std::abs(E[j])),
                static_cast<size_t>(std::abs(tau[j])),
                static_cast<size_t>(std::abs(style[j]))
            );

        // Convert continuous embedding -> symbolic patterns
        pm[j] = SymDync::GenSymbolicPattern(
            embeddings,
            relative,
            na_rm
        );
    }

    // Construct variable index vector for JE
    //
    // pm layout:
    //   pm[0], pm[1], ..., pm[n_vars-1]
    //
    // Therefore JE variables must be:
    //   {0,1,...,n_vars-1}
    std::vector<size_t> je_vars(n_vars);
    for (size_t i = 0; i < n_vars; ++i)
    {
        je_vars[i] = i;
    }

    // Compute Joint Entropy
    return InfoTheo::JE(pm, je_vars, base, na_rm);
}

// Wrapper function to calculate pattern entropy for time series data 
// [[Rcpp::export(rng = false)]]
double RcppSPE4TS(
    const Rcpp::NumericMatrix& mat,
    const Rcpp::IntegerVector& vars,
    const Rcpp::IntegerVector& E,
    const Rcpp::IntegerVector& tau,
    const Rcpp::IntegerVector& style,
    bool relative = true,
    double base = 2.0,
    bool na_rm = true
) {
    // Basic matrix dimensions
    const size_t n_cols = static_cast<size_t>(mat.ncol());
    const size_t n_obs  = static_cast<size_t>(mat.nrow());

    // Convert R variable indices -> C++ (0-based)
    std::vector<size_t> v = Rcpp::as<std::vector<size_t>>(vars);
    for (auto& idx : v)
    {
        if (idx < 1 || idx > n_cols)
        {
            Rcpp::stop(
                "Column index %d out of bounds [1, %d]",
                static_cast<int>(idx),
                static_cast<int>(n_cols)
            );
        }

        idx -= 1;  // convert to 0-based indexing
    }

    const size_t n_vars = v.size();

    if (n_vars == 0 || n_obs == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Pattern matrix
    std::vector<std::vector<std::vector<uint8_t>>> pm;
    pm.resize(n_vars);

    // Process each selected variable
    for (size_t j = 0; j < n_vars; ++j)
    {
        size_t col_id = v[j];

        // Extract column vector from R matrix
        std::vector<double> vec(n_obs);
        for (size_t r = 0; r < n_obs; ++r)
        {
            vec[r] = mat(r, col_id);
        }

        // Generate temporal embedding
        std::vector<std::vector<double>> embeddings =
            Embed::GenTSEmbedding(
                vec,
                static_cast<size_t>(std::abs(E[j])),
                static_cast<size_t>(std::abs(tau[j])),
                static_cast<size_t>(std::abs(style[j]))
            );

        // Convert continuous embedding -> symbolic patterns
        pm[j] = SymDync::GenSymbolicPattern(
            embeddings,
            relative,
            na_rm
        );
    }

    // Construct variable index vector for JE
    //
    // pm layout:
    //   pm[0], pm[1], ..., pm[n_vars-1]
    //
    // Therefore JE variables must be:
    //   {0,1,...,n_vars-1}
    std::vector<size_t> je_vars(n_vars);
    for (size_t i = 0; i < n_vars; ++i)
    {
        je_vars[i] = i;
    }

    // Compute Joint Entropy
    return InfoTheo::JE(pm, je_vars, base, na_rm);
}

// Wrapper function to calculate pattern mutual information for spatial lattice data 
// [[Rcpp::export(rng = false)]]
double RcppSPMI4Lattice(
    const Rcpp::NumericMatrix& mat,
    const Rcpp::IntegerVector& target,
    const Rcpp::IntegerVector& interact,
    const Rcpp::List& nb,
    const Rcpp::IntegerVector& E,
    const Rcpp::IntegerVector& tau,
    const Rcpp::IntegerVector& style,
    bool relative = true,
    double base = 2.0,
    bool na_rm = true
) {
    // Basic matrix dimensions
    const size_t n_cols = static_cast<size_t>(mat.ncol());
    const size_t n_obs  = static_cast<size_t>(mat.nrow());

    // Convert R variable indices -> C++ (0-based)
    std::vector<size_t> tv = Rcpp::as<std::vector<size_t>>(target);
    for (auto& idx : tv) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Target index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }

    std::vector<size_t> iv = Rcpp::as<std::vector<size_t>>(interact);
    for (auto& idx : iv) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Interact index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }

    std::vector<size_t> vars = tv;
    vars.insert(vars.end(), iv.begin(), iv.end());
    const size_t n_vars = vars.size();

    if (n_vars == 0 || n_obs == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Convert R neighbor structure -> std::vector<std::vector<size_t>>
    std::vector<std::vector<size_t>> nb_std = nb2std(nb);

    // Pattern matrix
    std::vector<std::vector<std::vector<uint8_t>>> pm;
    pm.resize(n_vars);

    // Process each selected variable
    for (size_t j = 0; j < n_vars; ++j)
    {
        size_t col_id = vars[j];

        // Extract column vector from R matrix
        std::vector<double> vec(n_obs);
        for (size_t r = 0; r < n_obs; ++r)
        {
            vec[r] = mat(r, col_id);
        }

        // Generate lattice embedding
        std::vector<std::vector<double>> embeddings =
            Embed::GenLatticeEmbedding(
                vec,
                nb_std,
                static_cast<size_t>(std::abs(E[j])),
                static_cast<size_t>(std::abs(tau[j])),
                static_cast<size_t>(std::abs(style[j]))
            );

        // Convert continuous embedding -> symbolic patterns
        pm[j] = SymDync::GenSymbolicPattern(
            embeddings,
            relative,
            na_rm
        );
    }

    // Construct variable index vector for MI
    std::vector<size_t> je_tv(tv.size());
    std::iota(je_tv.begin(), je_tv.end(), 0);

    std::vector<size_t> je_iv(iv.size());
    std::iota(je_iv.begin(), je_iv.end(), tv.size());

    // Compute mutual information
    return InfoTheo::MI(pm, je_tv, je_iv, base, na_rm);
}

// Wrapper function to calculate pattern mutual information for spatial grid data 
// [[Rcpp::export(rng = false)]]
double RcppSPMI4Grid(
    const Rcpp::NumericMatrix& mat,
    const Rcpp::IntegerVector& target,
    const Rcpp::IntegerVector& interact,
    const Rcpp::IntegerVector& E,
    const Rcpp::IntegerVector& tau,
    const Rcpp::IntegerVector& style,
    int nrows,
    bool relative = true,
    double base = 2.0,
    bool na_rm = true
) {
    // Basic matrix dimensions
    const size_t n_cols = static_cast<size_t>(mat.ncol());
    const size_t n_obs  = static_cast<size_t>(mat.nrow());

    // Convert R variable indices -> C++ (0-based)
    std::vector<size_t> tv = Rcpp::as<std::vector<size_t>>(target);
    for (auto& idx : tv) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Target index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }

    std::vector<size_t> iv = Rcpp::as<std::vector<size_t>>(interact);
    for (auto& idx : iv) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Interact index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }

    std::vector<size_t> vars = tv;
    vars.insert(vars.end(), iv.begin(), iv.end());
    const size_t n_vars = vars.size();

    if (n_vars == 0 || n_obs == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Pattern matrix
    std::vector<std::vector<std::vector<uint8_t>>> pm;
    pm.resize(n_vars);

    // Process each selected variable
    for (size_t j = 0; j < n_vars; ++j)
    {   
        size_t col_id = vars[j];

        // Extract subset matrix from R matrix
        std::vector<std::vector<double>> cm(
            nrows, std::vector<double>(n_obs / nrows));
        for (size_t r = 0; r < n_obs; ++r)
        {
            cm[r % nrows][r / nrows] = mat(r, col_id);
        }

        // Generate grid embedding
        std::vector<std::vector<double>> embeddings =
            Embed::GenGridEmbedding(
                cm,
                static_cast<size_t>(std::abs(E[j])),
                static_cast<size_t>(std::abs(tau[j])),
                static_cast<size_t>(std::abs(style[j]))
            );

        // Convert continuous embedding -> symbolic patterns
        pm[j] = SymDync::GenSymbolicPattern(
            embeddings,
            relative,
            na_rm
        );
    }

    // Construct variable index vector for MI
    std::vector<size_t> je_tv(tv.size());
    std::iota(je_tv.begin(), je_tv.end(), 0);

    std::vector<size_t> je_iv(iv.size());
    std::iota(je_iv.begin(), je_iv.end(), tv.size());

    // Compute mutual information
    return InfoTheo::MI(pm, je_tv, je_iv, base, na_rm);
}

// Wrapper function to calculate pattern mutual information for time series data 
// [[Rcpp::export(rng = false)]]
double RcppSPMI4TS(
    const Rcpp::NumericMatrix& mat,
    const Rcpp::IntegerVector& target,
    const Rcpp::IntegerVector& interact,
    const Rcpp::IntegerVector& E,
    const Rcpp::IntegerVector& tau,
    const Rcpp::IntegerVector& style,
    bool relative = true,
    double base = 2.0,
    bool na_rm = true
) {
    // Basic matrix dimensions
    const size_t n_cols = static_cast<size_t>(mat.ncol());
    const size_t n_obs  = static_cast<size_t>(mat.nrow());

    // Convert R variable indices -> C++ (0-based)
    std::vector<size_t> tv = Rcpp::as<std::vector<size_t>>(target);
    for (auto& idx : tv) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Target index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }

    std::vector<size_t> iv = Rcpp::as<std::vector<size_t>>(interact);
    for (auto& idx : iv) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Interact index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }

    std::vector<size_t> vars = tv;
    vars.insert(vars.end(), iv.begin(), iv.end());
    const size_t n_vars = vars.size();

    if (n_vars == 0 || n_obs == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Pattern matrix
    std::vector<std::vector<std::vector<uint8_t>>> pm;
    pm.resize(n_vars);

    // Process each selected variable
    for (size_t j = 0; j < n_vars; ++j)
    {
        size_t col_id = vars[j];

        // Extract column vector from R matrix
        std::vector<double> vec(n_obs);
        for (size_t r = 0; r < n_obs; ++r)
        {
            vec[r] = mat(r, col_id);
        }

        // Generate temporal embedding
        std::vector<std::vector<double>> embeddings =
            Embed::GenTSEmbedding(
                vec,
                static_cast<size_t>(std::abs(E[j])),
                static_cast<size_t>(std::abs(tau[j])),
                static_cast<size_t>(std::abs(style[j]))
            );

        // Convert continuous embedding -> symbolic patterns
        pm[j] = SymDync::GenSymbolicPattern(
            embeddings,
            relative,
            na_rm
        );
    }

    // Construct variable index vector for MI
    std::vector<size_t> je_tv(tv.size());
    std::iota(je_tv.begin(), je_tv.end(), 0);

    std::vector<size_t> je_iv(iv.size());
    std::iota(je_iv.begin(), je_iv.end(), tv.size());

    // Compute mutual information
    return InfoTheo::MI(pm, je_tv, je_iv, base, na_rm);
}
