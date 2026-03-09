/***************************************************************
 * File: combn.hpp
 *
 * Combinatorial utilities for generating combinations and subsets.
 *
 * Provides helper template functions for:
 *   - Generating all m-combinations from a given vector.
 *   - Generating all non-empty subsets of a vector.
 *
 * Implemented using recursive backtracking with minimal dependencies.
 * Suitable for general-purpose combinatorial enumeration tasks.
 *
 *  Author: Wenbo Lyu (Github: @SpatLyu)
 *  License: GPL-3
 */

#ifndef COMBN_HPP
#define COMBN_HPP

#include <vector>
#include <functional>

namespace Combn
{
    // ==============================
    // Combination generation
    // ==============================

    /**
     * @brief Generate all combinations of m elements from a given vector.
     *
     * Uses recursive backtracking to enumerate all size-m subsets
     * while preserving the original element order.
     *
     * @tparam T Element type
     * @param vec Input vector
     * @param m Number of elements per combination
     * @return std::vector<std::vector<T>> All possible m-combinations
     */
    template <typename T>
    inline std::vector<std::vector<T>> Combn(const std::vector<T>& vec, int m)
    {
        std::vector<std::vector<T>> result;
        std::vector<T> current;

        const int vec_size = static_cast<int>(vec.size());

        std::function<void(int)> helper = [&](int start)
        {
            if (static_cast<int>(current.size()) == m)
            {
                result.push_back(current);
                return;
            }

            int remaining = m - static_cast<int>(current.size());

            for (int i = start; i <= vec_size - remaining; ++i)
            {
                current.push_back(vec[i]);
                helper(i + 1);
                current.pop_back();
            }
        };

        helper(0);
        return result;
    }

    // ==============================
    // Subset generation
    // ==============================

    /**
     * @brief Generate all non-empty subsets of a given vector.
     *
     * Iteratively calls Combn for sizes 1 to n,
     * where n is the size of the input vector.
     *
     * @tparam T Element type
     * @param vec Input vector
     * @return std::vector<std::vector<T>> All non-empty subsets
     */
    template <typename T>
    inline std::vector<std::vector<T>> GenSubsets(const std::vector<T>& vec)
    {
        std::vector<std::vector<T>> allSubsets;

        for (int m = 1; m <= static_cast<int>(vec.size()); ++m)
        {
            std::vector<std::vector<T>> combs = Combn(vec, m);
            allSubsets.insert(allSubsets.end(), combs.begin(), combs.end());
        }

        return allSubsets;
    }

} // namespace Combn

#endif // COMBN_HPP
