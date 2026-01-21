/***************************************************************
 *
 *  File: symdync.hpp
 *
 *  Symbolic Dynamics Utilities for High Performance
 *  Pattern Construction and Encoding.
 *
 *  This header provides lightweight and efficient utilities
 *  for transforming continuous state space data into symbolic
 *  pattern representations suitable for large scale causal,
 *  information theoretic and dynamical systems analysis.
 *
 *  Core functionalities:
 *
 *    1. GenSignatureSpace
 *       - Converts a continuous state space matrix into a
 *         signature space matrix by computing successive
 *         differences or relative changes.
 *
 *    2. GenPatternSpace
 *       - Converts a continuous signature matrix into compact
 *         discrete pattern representations using uint8 encoding.
 *
 *  Data conventions:
 *
 *    Signature matrix:
 *      std::vector<std::vector<double>>
 *      Dimension: [n_rows x n_cols]
 *
 *    Pattern representation:
 *      std::vector<std::vector<uint8_t>>
 *      Dimension: [n_rows x pattern_length]
 *
 *    Symbol encoding:
 *      0  -> NA / undefined
 *      1  -> Downward change
 *      2  -> No change / Stable
 *      3  -> Upward change
 *
 *  NA handling:
 *
 *    Controlled by the parameter NA_rm.
 *
 *      NA_rm = true
 *        Rows containing any NaN are replaced by a single-element
 *        pattern {0}, indicating invalid observation.
 *
 *      NA_rm = false
 *        NaN values are encoded explicitly as symbol 0 inside the
 *        pattern vector.
 *
 *  Design principles:
 *
 *    - Memory efficiency
 *      Uses uint8 storage instead of string representations to
 *      minimize memory footprint and allocator overhead.
 *
 *    - Cache friendliness
 *      Patterns are stored in contiguous memory layouts for
 *      high throughput numerical processing.
 *
 *    - Performance scalability
 *      Suitable for millions of observations and seamless
 *      integration with high performance hashing and bit packing
 *      pipelines.
 *
 *    - Interoperability
 *      Pattern encoding is directly compatible with downstream
 *      information theoretic estimators and symbolic causal models.
 *
 *  Intended usage:
 *
 *    - Symbolic dynamics
 *    - Spatio temporal pattern mining
 *    - Information theoretic analysis
 *    - Causal pattern discovery
 *
 *  Author: wenbo lv
 *  License: GPL-3
 *
 ***************************************************************/

#ifndef SYMDYNC_HPP
#define SYMDYNC_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include "numericutils.hpp"

namespace SymDync
{

/**
 * @brief Computes the Signature Space Matrix from a State Space Matrix.
 *
 * This function transforms a state space matrix into a signature space matrix by
 * computing the differences between successive elements in each row. The transformation
 * captures dynamic patterns in state space.
 *
 * For each row in the input matrix:
 * - If relative == true, computes relative changes: (x[i+1] - x[i]) / x[i]
 * - If relative == false, computes absolute changes: x[i+1] - x[i]
 *
 * The output matrix has the same number of rows as the input, but the number of columns
 * is reduced by one.
 *
 * Special handling:
 * - Input validation (non-empty, at least 2 columns)
 * - When the difference is numerically zero, the signature value is set to 0.0
 * - NaN values propagate naturally
 *
 * @param mat       Input state space matrix [n_rows x n_cols].
 * @param relative  If true, compute relative change, otherwise absolute change.
 * @return          Signature space matrix [n_rows x (n_cols - 1)].
 * @throws std::invalid_argument if input is empty or has fewer than 2 columns.
 */
inline std::vector<std::vector<double>> GenSignatureSpace(
    const std::vector<std::vector<double>>& mat,
    bool relative = true
) {
    if (mat.empty()) {
        throw std::invalid_argument("Input matrix must not be empty.");
    }

    const size_t n_rows = mat.size();
    const size_t n_cols = mat[0].size();

    if (n_cols < 2) {
        throw std::invalid_argument("State space matrix must have at least 2 columns.");
    }

    const size_t out_cols = n_cols - 1;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // Pre-allocate output matrix filled with NaN
    std::vector<std::vector<double>> result(
        n_rows, std::vector<double>(out_cols, nan)
    );

    for (size_t i = 0; i < n_rows; ++i) {
        const auto& row = mat[i];
        auto& out_row = result[i];

        for (size_t j = 0; j < out_cols; ++j) {
            double diff = row[j + 1] - row[j];

            // NaN diff remains NaN
            if (!std::isnan(diff)) {
                if (NumericUtils::doubleNearlyEqual(diff, 0.0)) {
                    out_row[j] = 0.0;
                } else if (relative) {
                    out_row[j] = diff / row[j];
                } else {
                    out_row[j] = diff;
                }
            }
        }
    }

    return result;
}


/**
 * @brief Converts a continuous signature space matrix into a discrete
 *        pattern representation using compact uint8 encoding.
 *
 * Each numerical signature value is mapped to a categorical symbol:
 *
 *   Encoding:
 *     0  -> NA / undefined (NaN)
 *     1  -> negative change   (value < 0)
 *     2  -> no change         (value == 0)
 *     3  -> positive change   (value > 0)
 *
 * Output type:
 *   std::vector<std::vector<uint8_t>>
 *   Each inner vector represents one pattern instance.
 *
 * Behavior controlled by NA_rm:
 *
 *   NA_rm = true (default)
 *     - If a row contains any NaN, the entire pattern is replaced by a
 *       single-element vector {0}.
 *
 *   NA_rm = false
 *     - All rows are encoded.
 *     - NaN values are encoded as 0 inside the pattern vector.
 *
 * Example:
 *
 *   Input row:
 *     [ 0.1, -0.2, 0.0, NaN ]
 *
 *   NA_rm = true  -> {0}
 *   NA_rm = false -> {3, 1, 2, 0}
 *
 * Design rationale:
 *   - uint8_t encoding is memory efficient and cache friendly.
 *   - Avoids string allocation and hashing overhead.
 *   - Directly compatible with high performance information theoretic
 *     estimators and bit packing pipelines.
 *
 * @param mat    Input signature matrix [n_rows x n_cols], may contain NaN.
 * @param NA_rm  Whether to remove rows containing NaN.
 * @return       Vector of encoded patterns.
 */
inline std::vector<std::vector<uint8_t>> GenPatternSpace(
    const std::vector<std::vector<double>>& mat,
    bool NA_rm = true
) {
    std::vector<std::vector<uint8_t>> patterns;
    if (mat.empty()) return patterns;

    const size_t n_rows = mat.size();
    const size_t n_cols = mat[0].size();
    patterns.reserve(n_rows);

    for (size_t i = 0; i < n_rows; ++i) {
        const auto& row = mat[i];

        bool has_nan = false;
        std::vector<uint8_t> pat;
        pat.reserve(n_cols);

        for (size_t j = 0; j < n_cols; ++j) {
            double v = row[j];

            if (std::isnan(v)) {
                has_nan = true;
                pat.push_back(static_cast<uint8_t>(0));
            }
            else if (NumericUtils::doubleNearlyEqual(v, 0.0)) {
                pat.push_back(static_cast<uint8_t>(2));
            }
            else if (v > 0.0) {
                pat.push_back(static_cast<uint8_t>(3));
            }
            else {
                pat.push_back(static_cast<uint8_t>(1));
            }
        }

        // NA handling
        if (NA_rm && has_nan) {
            patterns.emplace_back(std::vector<uint8_t>{0});
        } else {
            patterns.emplace_back(std::move(pat));
        }
    }

    return patterns;
}

} // namespace SymDync

#endif // SYMDYNC_HPP
