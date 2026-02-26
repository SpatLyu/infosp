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
std::vector<std::vector<size_t>> nb2std(const Rcpp::List& nb);

// Function to convert std::vector<std::vector<size_t>> (the `nb` object) to Rcpp::List
Rcpp::List std2nb(const std::vector<std::vector<size_t>>& nb);

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
std::vector<uint8_t> index2base4(uint64_t idx);

/********************************************************************
 *  vec2pat
 *
 *  Convert an R vector (Integer / Numeric / Character)
 *  into std::vector<std::vector<uint8_t>>.
 *
 *  Design:
 *      - Extract unique non NA values
 *      - Sort from low to high
 *      - Assign increasing index 0,1,2,...
 *      - Encode index in base 4
 *      - Each observation becomes one Pattern
 *
 *  Result:
 *      PatternSeries of size N (std::vector<std::vector<uint8_t>>)
 *      Each Pattern contains base 4 digits representing category
 *
 *  NA handling:
 *      NA is encoded as single digit {0}
 *
 ********************************************************************/

// -------- IntegerVector --------
std::vector<std::vector<uint8_t>> vec2pat(const Rcpp::IntegerVector& v);

// -------- NumericVector --------
std::vector<std::vector<uint8_t>> vec2pat(const Rcpp::NumericVector& v);

// -------- CharacterVector --------
std::vector<std::vector<uint8_t>> vec2pat(const Rcpp::CharacterVector& v);

/********************************************************************
 *  mat2patmat
 *
 *  Convert an R matrix (Integer / Numeric / Character)
 *  into std::vector<std::vector<std::vector<uint8_t>>>.
 *
 *  Structure:
 *
 *      R matrix:
 *          n rows  = observations
 *          p cols  = variables
 *
 *      C++ std::vector<std::vector<std::vector<uint8_t>>>:
 *          Matrix[var][obs]
 *
 *  Each column of the R matrix is converted using vec2pat,
 *  producing a PatternSeries (std::vector<std::vector<uint8_t>>). 
 *  The PatternSeries objects are stored sequentially into the 
 *  Matrix container (std::vector<std::vector<std::vector<uint8_t>>>).
 *
 *  NA handling:
 *      Delegated to vec2pat.
 *
 ********************************************************************/
std::vector<std::vector<std::vector<uint8_t>>> mat2patmat(SEXP x);

/********************************************************************
 *  pat2vec
 *
 *  Convert a PatternSeries into an Rcpp::CharacterVector.
 *
 *  Input:
 *      std::vector<std::vector<uint8_t>>
 *
 *      Each inner std::vector<uint8_t> represents one symbolic
 *      pattern. Each element is a base-4 digit (0,1,2,3).
 *
 *  Conversion rule:
 *      - Each uint8_t digit is converted to its ASCII character.
 *      - Digits are concatenated without separator.
 *      - The entire pattern becomes one std::string.
 *      - Each pattern string becomes one element of the
 *        resulting CharacterVector.
 *
 *  Example:
 *      { {3,1,2,0}, {1}, {0} }
 *
 *      becomes
 *
 *      c("3120", "1", "0")
 *
 *  Notes:
 *      - Order is preserved.
 *      - No delimiter is inserted.
 *      - Empty input returns empty CharacterVector.
 *
 ********************************************************************/
Rcpp::CharacterVector
pat2vec(const std::vector<std::vector<uint8_t>>& pat);

/********************************************************************
 *  patmat2mat
 *
 *  Convert a PatternMatrix into an Rcpp::CharacterMatrix.
 *
 *  Input structure:
 *
 *      std::vector<std::vector<std::vector<uint8_t>>>
 *
 *      Interpreted as:
 *
 *          patmat[var][obs]
 *
 *      Outer dimension  = variables (columns)
 *      Inner dimension  = observations (rows)
 *
 *  Output structure:
 *
 *      Rcpp::CharacterMatrix
 *
 *          n rows = observations
 *          p cols = variables
 *
 *  Conversion rule:
 *      - Each pattern (std::vector<uint8_t>) is converted to a
 *        string exactly as in pat2vec.
 *      - Stored at position (obs, var).
 *
 *  Example:
 *
 *      patmat[0] = PatternSeries of variable 1
 *      patmat[1] = PatternSeries of variable 2
 *
 *      Result:
 *          matrix where column j corresponds to patmat[j]
 *
 *  Requirements:
 *      - All PatternSeries must have equal length.
 *      - Throws std::invalid_argument if inconsistent.
 *
 ********************************************************************/
Rcpp::CharacterMatrix
patmat2mat(const std::vector<std::vector<std::vector<uint8_t>>>& patmat);

#endif // DataTrans_H
