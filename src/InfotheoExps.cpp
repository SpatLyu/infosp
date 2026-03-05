#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <limits>
#include <numeric>
#include <algorithm>
#include "infotheo.hpp"
#include "surd.hpp"
#include "DataTrans.h"

// Wrapper function to calculate Shannon entropy
// [[Rcpp::export(rng = false)]]
double RcppEntropy(SEXP series,
                   double base = 2.0,
                   bool na_rm = true)
{
    InfoTheo::PatternSeries s;

    switch (TYPEOF(series))
    {
        case INTSXP:
        {
            Rcpp::IntegerVector v(series);
            s = vec2pat(v);
            break;
        }

        case REALSXP:
        {
            Rcpp::NumericVector v(series);
            s = vec2pat(v);
            break;
        }

        case STRSXP:
        {
            Rcpp::CharacterVector v(series);
            s = vec2pat(v);
            break;
        }

        default:
            Rcpp::stop("Input must be Integer, Numeric, or Character vector.");
    }

    return InfoTheo::Entropy(s, base, na_rm);
}

// Wrapper function to calculate joint entropy
// [[Rcpp::export(rng = false)]]
double RcppJE(SEXP mat,
              const Rcpp::IntegerVector& vars,
              double base = 2.0,
              bool na_rm = true)
{
    InfoTheo::Matrix m = mat2patmat(mat);
    std::vector<size_t> v = Rcpp::as<std::vector<size_t>>(vars);

    const size_t n_cols = m.size();
    for (auto& idx : v) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Column index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }
    
    return InfoTheo::JE(m, v, base, na_rm);
}

// Wrapper function to calculate conditional entropy
// [[Rcpp::export(rng = false)]]
double RcppCE(SEXP mat,
              const Rcpp::IntegerVector& target,
              const Rcpp::IntegerVector& conds,
              double base = 2.0,
              bool na_rm = true)
{
    InfoTheo::Matrix m = mat2patmat(mat);

    std::vector<size_t> t = Rcpp::as<std::vector<size_t>>(target);
    std::vector<size_t> c = Rcpp::as<std::vector<size_t>>(conds);

    const size_t n_cols = m.size();
    for (auto& idx : t) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Target index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }
    for (auto& idx : c) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Conds index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }

    return InfoTheo::CE(m, t, c, base, na_rm);
}

// Wrapper function to calculate mutual information
// [[Rcpp::export(rng = false)]]
double RcppMI(SEXP mat,
              const Rcpp::IntegerVector& target,
              const Rcpp::IntegerVector& interact,
              double base = 2.0,
              bool na_rm = true)
{
    InfoTheo::Matrix m = mat2patmat(mat);

    std::vector<size_t> t = Rcpp::as<std::vector<size_t>>(target);
    std::vector<size_t> i = Rcpp::as<std::vector<size_t>>(interact);

    const size_t n_cols = m.size();
    for (auto& idx : t) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Target index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }
    for (auto& idx : i) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Interact index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }

    return InfoTheo::MI(m, t, i, base, na_rm);
}

// Wrapper function to calculate conditional mutual information
// [[Rcpp::export(rng = false)]]
double RcppCMI(SEXP mat,
               const Rcpp::IntegerVector& target,
               const Rcpp::IntegerVector& interact,
               const Rcpp::IntegerVector& conds,
               double base = 2.0,
               bool na_rm = true)
{
    InfoTheo::Matrix m = mat2patmat(mat);

    std::vector<size_t> t = Rcpp::as<std::vector<size_t>>(target);
    std::vector<size_t> i = Rcpp::as<std::vector<size_t>>(interact);
    std::vector<size_t> c = Rcpp::as<std::vector<size_t>>(conds);

    const size_t n_cols = m.size();
    for (auto& idx : t) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Target index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }
    for (auto& idx : i) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Interact index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }
    for (auto& idx : c) {
        if (idx < 1 || idx > n_cols) {
            Rcpp::stop("Conds index %d out of bounds [1, %d]", 
                       static_cast<int>(idx), 
                       static_cast<int>(n_cols));
        }
        idx -= 1;  // to 0-based
    }

    return InfoTheo::CMI(m, t, i, c, base, na_rm);
}

// Wrapper function to calculate SURD decomposition
// [[Rcpp::export(rng = false)]]
Rcpp::List RcppSURD(SEXP mat,
                    double base = 2.0,
                    bool na_rm = true,
                    bool normalize = false,
                    int threads = 1)
{
    InfoTheo::Matrix m = mat2patmat(mat);

    SURD::SURDRes res =
        SURD::SURD(m, base, na_rm, normalize, static_cast<size_t>(std::abs(threads)));

    const size_t k = res.size();

    Rcpp::NumericVector values(k);
    Rcpp::CharacterVector types(k);
    Rcpp::CharacterVector names(k);

    for (size_t i = 0; i < k; ++i)
    {
        values[i] = res.values[i];

        // variable name
        if (res.types[i] == 3)
        {
            // InfoLeak uses all sources
            std::string nm = "InfoLeak";
            names[i] = nm;
            types[i] = "InfoLeak";
            continue;
        }

        const auto& vars = res.var_indices[i];

        std::string nm;

        for (size_t j = 0; j < vars.size(); ++j)
        {
            if (j > 0)
                nm += "_";

            nm += "V";
            nm += std::to_string(vars[j]);
        }

        names[i] = nm;

        switch (res.types[i])
        {
            case 0:
                types[i] = "R";
                break;
            case 1:
                types[i] = "U";
                break;
            case 2:
                types[i] = "S";
                break;
            default:
                types[i] = "Unknown";
        }
    }

    values.attr("names") = names;

    return Rcpp::List::create(
        Rcpp::Named("values") = values,
        Rcpp::Named("types")  = types
    );
}
