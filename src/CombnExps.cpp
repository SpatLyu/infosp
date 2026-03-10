#include <vector>
#include <string>
#include "combn.hpp"
#include <Rcpp.h>

// Helper: convert C++ vector-of-vectors result to Rcpp::List
template <typename T>
Rcpp::List convert2RList(const std::vector<std::vector<T>>& cpp_result) {
    Rcpp::List out(cpp_result.size());
    for (size_t i = 0; i < cpp_result.size(); ++i) {
        out[i] = Rcpp::wrap(cpp_result[i]);
    }
    return out;
}

// Wrapper function to generate all combinations of m elements from an integer, numeric, or character vector.
// [[Rcpp::export(rng = false)]]
Rcpp::List RcppCombn(SEXP vec, int m) {
    switch (TYPEOF(vec)) {
        case INTSXP: {
            Rcpp::IntegerVector vec_int(vec);
            std::vector<int> cpp_vec(vec_int.begin(), vec_int.end());
            auto result = Combn::Combn(cpp_vec, static_cast<size_t>(m));
            return convert2RList(result);
        }
        case REALSXP: {
            Rcpp::NumericVector vec_num(vec);
            std::vector<double> cpp_vec(vec_num.begin(), vec_num.end());
            auto result = Combn::Combn(cpp_vec, static_cast<size_t>(m));
            return convert2RList(result);
        }
        case STRSXP: {
            Rcpp::CharacterVector vec_char(vec);
            std::vector<std::string> cpp_vec = Rcpp::as<std::vector<std::string>>(vec_char);
            auto result = Combn::Combn(cpp_vec, static_cast<size_t>(m));
            return convert2RList(result);
        }
        default:
            Rcpp::stop("vec must be an integer, numeric, or character vector");
    }
}

// Wrapper function to generate all non-empty subsets from an integer, numeric, or character vector.
// [[Rcpp::export(rng = false)]]
Rcpp::List RcppGenSubsets(SEXP vec) {
    switch (TYPEOF(vec)) {
        case INTSXP: {
            Rcpp::IntegerVector vec_int(vec);
            std::vector<int> cpp_vec(vec_int.begin(), vec_int.end());
            auto result = Combn::GenSubsets(cpp_vec);
            return convert2RList(result);
        }
        case REALSXP: {
            Rcpp::NumericVector vec_num(vec);
            std::vector<double> cpp_vec(vec_num.begin(), vec_num.end());
            auto result = Combn::GenSubsets(cpp_vec);
            return convert2RList(result);
        }
        case STRSXP: {
            Rcpp::CharacterVector vec_char(vec);
            std::vector<std::string> cpp_vec = Rcpp::as<std::vector<std::string>>(vec_char);
            auto result = Combn::GenSubsets(cpp_vec);
            return convert2RList(result);
        }
        default:
            Rcpp::stop("vec must be an integer, numeric, or character vector");
    }
}
