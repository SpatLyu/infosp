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
std::vector<std::vector<size_t>> nb2std(const Rcpp::List& nb) {
  // Get the number of elements in the nb object
  size_t n = static_cast<size_t>(nb.size());
  if (n <= 1) {
    Rcpp::stop("The nb object must contain at least two spatial units (got %d)", n);
  }
  
  // Create a std::vector<std::vector<size_t>> to store the result
  std::vector<std::vector<size_t>> result(n);

  // Iterate over each element in the nb object
  for (size_t i = 0; i < n; ++i) {
    // Get the current element (should be an integer vector)
    Rcpp::IntegerVector current_nb = nb[i];
    size_t cur_num_nb = static_cast<size_t>(current_nb.size());

    // Create a vector<size_t> to store the current subset of elements
    std::vector<size_t> current_subset;
    current_subset.reserve(cur_num_nb);

    // Iterate over each element in the current subset
    for (size_t j = 0; j < cur_num_nb; ++j) {
      // Subtract one from each element to convert from R's 1-based indexing to C++'s 0-based indexing
      current_subset.push_back(current_nb[j] - 1);
    }

    // Add the current subset to the result
    result[i] = current_subset;
  }

  return result;
}

// Function to convert std::vector<std::vector<size_t>> (the `nb` object) to Rcpp::List
Rcpp::List std2nb(const std::vector<std::vector<size_t>>& nb) {
  size_t n = nb.size();
  Rcpp::List result(n);

  for (size_t i = 0; i < n; ++i) {
    const auto& neighbors = nb[i];
    Rcpp::IntegerVector r_neighbors(neighbors.size());
    for (size_t j = 0; j < neighbors.size(); ++j) {
      r_neighbors[j] = static_cast<int>(neighbors[j] + 1);
    }
    result[i] = r_neighbors;
  }

  return result;
}

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
std::vector<uint8_t> index2base4(uint64_t idx)
{
    std::vector<uint8_t> digits;

    if (idx == 0)
    {
        digits.push_back(0);
        return digits;
    }

    while (idx > 0)
    {
        digits.push_back(static_cast<uint8_t>(idx & 0x3)); // idx % 4
        idx >>= 2;                                        // idx /= 4
    }

    return digits; // little endian base 4
}

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
std::vector<std::vector<uint8_t>> vec2pat(const Rcpp::IntegerVector& v)
{
    std::vector<std::vector<uint8_t>> series;
    series.reserve(v.size());

    // 1. Collect unique non NA values
    std::vector<int> uniq;
    uniq.reserve(v.size());

    for (int val : v)
        if (!Rcpp::IntegerVector::is_na(val))
            uniq.push_back(val);

    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    // 2. Map value → index
    std::unordered_map<int, uint64_t> dict;
    for (uint64_t i = 0; i < uniq.size(); ++i)
        dict[uniq[i]] = i;

    // 3. Encode each observation
    for (int val : v)
    {
        if (Rcpp::IntegerVector::is_na(val))
        {
            series.push_back( std::vector<uint8_t>{0} );
        }
        else
        {
            uint64_t idx = dict[val];
            series.push_back( index2base4(idx) );
        }
    }

    return series;
}

// -------- NumericVector --------
std::vector<std::vector<uint8_t>> vec2pat(const Rcpp::NumericVector& v)
{
    std::vector<std::vector<uint8_t>> series;
    series.reserve(v.size());

    std::vector<double> uniq;
    uniq.reserve(v.size());

    for (double val : v)
        if (!Rcpp::NumericVector::is_na(val))
            uniq.push_back(val);

    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    std::unordered_map<double, uint64_t> dict;
    for (uint64_t i = 0; i < uniq.size(); ++i)
        dict[uniq[i]] = i;

    for (double val : v)
    {
        if (Rcpp::NumericVector::is_na(val))
        {
            series.push_back( std::vector<uint8_t>{0} );
        }
        else
        {
            uint64_t idx = dict[val];
            series.push_back( index2base4(idx) );
        }
    }

    return series;
}

// -------- CharacterVector --------
std::vector<std::vector<uint8_t>> vec2pat(const Rcpp::CharacterVector& v)
{
    std::vector<std::vector<uint8_t>> series;
    series.reserve(v.size());

    std::vector<std::string> uniq;
    uniq.reserve(v.size());

    for (Rcpp::String s : v)
        if (s != NA_STRING)
            uniq.push_back(std::string(s));

    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    std::unordered_map<std::string, uint64_t> dict;
    for (uint64_t i = 0; i < uniq.size(); ++i)
        dict[uniq[i]] = i;

    for (Rcpp::String s : v)
    {
        if (s == NA_STRING)
        {
            series.push_back( std::vector<uint8_t>{0} );
        }
        else
        {
            uint64_t idx = dict[std::string(s)];
            series.push_back( index2base4(idx) );
        }
    }

    return series;
}

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
std::vector<std::vector<std::vector<uint8_t>>> mat2patmat(SEXP x)
{
    if (!Rf_isMatrix(x))
        Rcpp::stop("Input must be a matrix.");

    std::vector<std::vector<std::vector<uint8_t>>> mat;

    switch (TYPEOF(x))
    {
        case INTSXP:
        {
            Rcpp::IntegerMatrix m(x);
            const size_t p = static_cast<size_t>(m.ncol());
            mat.reserve(p);

            for (size_t j = 0; j < p; ++j)
            {
                Rcpp::IntegerVector col = m(Rcpp::_, j);
                mat.push_back(vec2pat(col));
            }

            break;
        }

        case REALSXP:
        {
            Rcpp::NumericMatrix m(x);
            const size_t p = static_cast<size_t>(m.ncol());
            mat.reserve(p);

            for (size_t j = 0; j < p; ++j)
            {
                Rcpp::NumericVector col = m(Rcpp::_, j);
                mat.push_back(vec2pat(col));
            }

            break;
        }

        case STRSXP:
        {
            Rcpp::CharacterMatrix m(x);
            const size_t p = static_cast<size_t>(m.ncol());
            mat.reserve(p);

            for (size_t j = 0; j < p; ++j)
            {
                Rcpp::CharacterVector col = m(Rcpp::_, j);
                mat.push_back(vec2pat(col));
            }

            break;
        }

        default:
            Rcpp::stop("Matrix must be Integer, Numeric, or Character.");
    }

    return mat;
}

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
pat2vec(const std::vector<std::vector<uint8_t>>& pat)
{
    const size_t n = pat.size();
    Rcpp::CharacterVector out(n);

    for (size_t i = 0; i < n; ++i)
    {
        const auto& pattern = pat[i];

        std::string s;
        s.reserve(pattern.size());

        for (uint8_t digit : pattern)
        {
            // digit is 0,1,2,3
            s.push_back(static_cast<char>('0' + digit));
        }

        out[i] = s;
    }

    return out;
}

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
patmat2mat(const std::vector<std::vector<std::vector<uint8_t>>>& patmat)
{
    if (patmat.empty())
        return Rcpp::CharacterMatrix(0, 0);

    const size_t p = patmat.size();              // number of variables
    const size_t n = patmat[0].size();           // number of observations

    // Validate equal lengths
    for (size_t j = 1; j < p; ++j)
    {
        if (patmat[j].size() != n)
        {
            throw std::invalid_argument(
                "All PatternSeries must have identical length."
            );
        }
    }

    Rcpp::CharacterMatrix out(n, p);

    for (size_t j = 0; j < p; ++j)
    {
        for (size_t i = 0; i < n; ++i)
        {
            const auto& pattern = patmat[j][i];

            std::string s;
            s.reserve(pattern.size());

            for (uint8_t digit : pattern)
            {
                s.push_back(static_cast<char>('0' + digit));
            }

            out(i, j) = s;
        }
    }

    return out;
}
