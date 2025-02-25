#ifndef DataTrans_H
#define DataTrans_H

#include <vector>
#include <cstdint>
#include <string>
#include <limits>
#include <numeric>
#include <algorithm>
#include <unordered_map> 
#include "infotheo.hpp"
#include <Rcpp.h>

/********************************************************************
 *
 *  Spatial Neighbour Structure Conversion Utilities
 *
 *  These functions convert between:
 *
 *      R representation:
 *          Rcpp::List
 *          Each element is an IntegerVector
 *          Indices are 1 based (R convention)
 *
 *      C++ representation:
 *          std::vector<std::vector<size_t>>
 *          Indices are 0 based (C++ convention)
 *
 *  This structure corresponds to the common "nb" object used in
 *  spatial statistics to represent adjacency lists.
 *
 *  Example in R:
 *
 *      nb[[1]] = c(2, 3)
 *      nb[[2]] = c(1)
 *      nb[[3]] = c(1)
 *
 *  Meaning:
 *      Spatial unit 1 is neighbor with 2 and 3
 *      Spatial unit 2 is neighbor with 1
 *      Spatial unit 3 is neighbor with 1
 *
 *  Conversion rules:
 *
 *      R → C++
 *          - Convert 1 based indices to 0 based
 *          - Store in std::vector<std::vector<size_t>>
 *
 *      C++ → R
 *          - Convert 0 based indices to 1 based
 *          - Return Rcpp::List of IntegerVector
 *
 *  Assumptions:
 *
 *      - Input R list must contain at least two spatial units
 *      - Each element of the list must be an IntegerVector
 *      - No structural validation of symmetry is performed
 *
 ********************************************************************/

// Function to convert Rcpp::List to std::vector<std::vector<size_t>> (the `nb` object)
inline std::vector<std::vector<size_t>> nb2std(const Rcpp::List& nb);

// Function to convert std::vector<std::vector<size_t>> (the `nb` object) to Rcpp::List
inline Rcpp::List std2nb(const std::vector<std::vector<size_t>>& nb);

/********************************************************************
 *  index2base4
 *
 *  Convert a non negative integer index into base 4 representation.
 *
 *  Each base 4 digit is stored in one uint8_t.
 *  Only the lowest 2 bits of each uint8_t are meaningful.
 *
 *  Example:
 *      idx = 6
 *      base 4 = 12
 *      stored as {2,1}  (little endian)
 *
 *  This design allows unlimited categorical cardinality while
 *  remaining fully compatible with the 2 bit packing scheme
 *  used inside InfoTheo.
 *
 ********************************************************************/
inline std::vector<uint8_t> index2base4(uint64_t idx);

/********************************************************************
 *  vec2pat
 *
 *  Convert an R vector (Integer / Numeric / Character)
 *  into InfoTheo::PatternSeries.
 *
 *  Design:
 *      - Extract unique non NA values
 *      - Sort from low to high
 *      - Assign increasing index 0,1,2,...
 *      - Encode index in base 4
 *      - Each observation becomes one Pattern
 *
 *  Result:
 *      PatternSeries of size N
 *      Each Pattern contains base 4 digits representing category
 *
 *  NA handling:
 *      NA is encoded as single digit {0}
 *
 ********************************************************************/

// -------- IntegerVector --------
inline InfoTheo::PatternSeries vec2pat(const Rcpp::IntegerVector& v);

// -------- NumericVector --------
inline InfoTheo::PatternSeries vec2pat(const Rcpp::NumericVector& v);

// -------- CharacterVector --------
inline InfoTheo::PatternSeries vec2pat(const Rcpp::CharacterVector& v);

#endif // DataTrans_H
