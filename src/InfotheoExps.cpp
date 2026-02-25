#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <limits>
#include <numeric>
#include <algorithm>
#include "infotheo.hpp"
#include "DataTrans.h"

// Wrapper function to calculate Shannon entropy
// [[Rcpp::export(rng = false)]]
double RcppEntropy(SEXP series,
                   double base = 2.0,
                   bool NA_rm = false)
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

    return InfoTheo::Entropy(s, base, NA_rm);
}

// Wrapper function to calculate joint entropy
// [[Rcpp::export(rng = false)]]
double RcppJE(SEXP mat,
              Rcpp::IntegerVector vars,
              double base = 2.0,
              bool NA_rm = false)
{
    InfoTheo::Matrix m = mat2patmat(mat);
    std::vector<size_t> v = Rcpp::as<std::vector<size_t>>(vars);
    return InfoTheo::JE(m, v, base, NA_rm);
}

// Wrapper function to calculate conditional entropy
// [[Rcpp::export(rng = false)]]
double RcppCE(SEXP mat,
              Rcpp::IntegerVector target,
              Rcpp::IntegerVector conds,
              double base = 2.0,
              bool NA_rm = false)
{
    InfoTheo::Matrix m = mat2patmat(mat);

    std::vector<size_t> t = Rcpp::as<std::vector<size_t>>(target);
    std::vector<size_t> c = Rcpp::as<std::vector<size_t>>(conds);

    return InfoTheo::CE(m, t, c, base, NA_rm);
}

// Wrapper function to calculate mutual information
// [[Rcpp::export(rng = false)]]
double RcppMI(SEXP mat,
              Rcpp::IntegerVector target,
              Rcpp::IntegerVector interact,
              double base = 2.0,
              bool NA_rm = false)
{
    InfoTheo::Matrix m = mat2patmat(mat);

    std::vector<size_t> t = Rcpp::as<std::vector<size_t>>(target);
    std::vector<size_t> i = Rcpp::as<std::vector<size_t>>(interact);

    return InfoTheo::MI(m, t, i, base, NA_rm);
}

// Wrapper function to calculate conditional mutual information
// [[Rcpp::export(rng = false)]]
double RcppCMI(SEXP mat,
               Rcpp::IntegerVector target,
               Rcpp::IntegerVector interact,
               Rcpp::IntegerVector conds,
               double base = 2.0,
               bool NA_rm = false)
{
    InfoTheo::Matrix m = mat2patmat(mat);

    std::vector<size_t> t = Rcpp::as<std::vector<size_t>>(target);
    std::vector<size_t> i = Rcpp::as<std::vector<size_t>>(interact);
    std::vector<size_t> c = Rcpp::as<std::vector<size_t>>(conds);

    return InfoTheo::CMI(m, t, i, c, base, NA_rm);
}
