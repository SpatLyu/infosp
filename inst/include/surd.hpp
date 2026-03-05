/**********************************************************************
 * File: surd.hpp
 *
 * High-performance implementation of
 * Synergistic-Unique-Redundant Decomposition (SURD).
 *
 * Optimization strategy:
 *
 *   - Bitmask encoding for variable subsets
 *   - Variable index lookup table
 *   - Popcount lookup table
 *   - Contiguous entropy cache (vector)
 *   - Parallel entropy computation
 *   - Vector-based grouping (avoid std::map)
 *
 * Bit layout:
 *
 *   bit 0     -> target variable
 *   bit 1..n  -> source variables
 *
 * Type encoding in SURDRes::types:
 *   0 = Redundant
 *   1 = Unique
 *   2 = Synergistic
 *   3 = Information loss
 *
 * Optional normalization rescales the decomposed information
 * components (redundant, unique, synergistic) into [0,1].
 *
 * Author: Wenbo Lyu (Github: @SpatLyu)
 * License: GPL-3
 **********************************************************************/

#ifndef SURD_HPP
#define SURD_HPP

#include <vector>
#include <thread>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "infotheo.hpp"
#include "numericutils.hpp"

namespace SURD
{
    using Matrix = InfoTheo::Matrix;

    /***********************************************************
    * Result structure
    ***********************************************************/
    struct SURDRes
    {
        std::vector<double> values;
        std::vector<uint8_t> types;
        std::vector<std::vector<size_t>> var_indices;

        size_t size() const noexcept { return values.size(); }
    };

    /***********************************************************
    * Build mask -> variable index table
    ***********************************************************/
    inline std::vector<std::vector<size_t>>
    build_mask_vars(size_t total_masks)
    {
        std::vector<std::vector<size_t>> table(total_masks);

        for (uint64_t mask = 1; mask < total_masks; ++mask)
        {
            std::vector<size_t> vars;

            size_t idx = 0;
            uint64_t m = mask;

            while (m)
            {
                if (m & 1ULL)
                    vars.push_back(idx);

                m >>= 1;
                ++idx;
            }

            table[mask] = std::move(vars);
        }

        return table;
    }

    /***********************************************************
    * Build popcount table
    ***********************************************************/
    inline std::vector<uint8_t>
    build_popcount(size_t total_masks)
    {
        std::vector<uint8_t> table(total_masks);

        for (uint64_t i = 1; i < total_masks; ++i)
            table[i] = table[i >> 1] + (i & 1);

        return table;
    }

    /***********************************************************
    * Entropy computation task
    ***********************************************************/
    inline void entropy_task(
        const Matrix& mat,
        const std::vector<std::vector<size_t>>& mask_vars,
        std::vector<double>& H,
        size_t start,
        size_t end,
        double base,
        bool na_rm)
    {
        for (size_t mask = start; mask < end; ++mask)
        {
            const auto& vars = mask_vars[mask];

            if (vars.empty())
                continue;

            H[mask] = InfoTheo::JE(mat, vars, base, na_rm);
        }
    }

    /***********************************************************
    * Precompute entropy table
    ***********************************************************/
    inline std::vector<double>
    precompute_entropy(
        const Matrix& mat,
        const std::vector<std::vector<size_t>>& mask_vars,
        double base,
        bool na_rm,
        size_t threads)
    {
        const size_t total_masks = mask_vars.size();

        std::vector<double> H(
            total_masks,
            std::numeric_limits<double>::quiet_NaN()
        );

        if (threads <= 1 || total_masks < 1024)
        {
            entropy_task(mat, mask_vars, H, 0, total_masks, base, na_rm);
            return H;
        }

        const size_t chunk = (total_masks + threads - 1) / threads;

        std::vector<std::thread> workers;

        for (size_t t = 0; t < threads; ++t)
        {
            size_t start = t * chunk;
            size_t end = std::min(start + chunk, total_masks);

            if (start >= end)
                break;

            workers.emplace_back(
                entropy_task,
                std::cref(mat),
                std::cref(mask_vars),
                std::ref(H),
                start,
                end,
                base,
                na_rm
            );
        }

        for (auto& th : workers)
            th.join();

        return H;
    }

    /***********************************************************
    * Compute mutual information
    ***********************************************************/
    inline double compute_mi(
        const std::vector<double>& H,
        uint64_t subset_mask)
    {
        constexpr uint64_t target_mask = 1ULL;

        uint64_t joint = subset_mask | target_mask;

        double ht  = H[target_mask];
        double hs  = H[subset_mask];
        double hts = H[joint];

        if (std::isnan(ht) || std::isnan(hs) || std::isnan(hts))
            return std::numeric_limits<double>::quiet_NaN();

        return ht + hs - hts;
    }

    /***********************************************************
    * SURD main algorithm
    ***********************************************************/
    inline SURDRes SURD(
        const Matrix& mat,
        double base = 2.0,
        bool na_rm = true,
        bool normalize = false,
        size_t threads = 1,
        size_t max_order = std::numeric_limits<size_t>::max())
    {   
        if (threads == 0) threads = 1;
        size_t hw = std::thread::hardware_concurrency();
        if (hw > 0) threads = std::min(threads, hw);

        SURDRes result;

        if (mat.size() < 2)
            return result;

        const size_t n_vars = mat.size();
        const size_t n_sources = mat.size() - 1;
        max_order = std::min(max_order, n_sources);

        if (n_vars >= 63)
            throw std::invalid_argument("SURD supports <63 variables");

        const uint64_t total_masks = (1ULL << n_vars);

        /***********************************************************
        * Build lookup tables
        ***********************************************************/
        auto mask_vars = build_mask_vars(total_masks);
        auto popcount  = build_popcount(total_masks);

        /***********************************************************
        * Precompute entropies
        ***********************************************************/
        auto H = precompute_entropy(mat, mask_vars, base, na_rm, threads);

        struct Entry
        {
            double mi;
            uint64_t mask;
            uint8_t order;
        };

        size_t approx = 0;
        if (n_sources > 30) {
            approx = total_masks / 2;
        } else {
            for (size_t k = 1; k <= max_order; ++k)
            {
                size_t c = 1;
                for (size_t i = 0; i < k; ++i)
                    c = c * (n_sources - i) / (i + 1);
                approx += c;
            }
        }
        
        std::vector<Entry> entries;
        entries.reserve(approx);

        const uint64_t start_mask = 2ULL;

        for (uint64_t mask = start_mask; mask < total_masks; ++mask)
        {
            if (mask & 1ULL)
                continue;

            uint8_t order = popcount[mask];
            if (static_cast<size_t>(order) > max_order)
                continue;

            double mi = compute_mi(H, mask);

            if (!std::isnan(mi) && mi >= -1e-12)
            {
                entries.push_back({
                    std::max(0.0, mi),
                    mask,
                    order
                });
            }
        }

        if (entries.empty())
            return result;

        /***********************************************************
        * Group by order
        ***********************************************************/
        std::vector<std::vector<Entry*>> groups(max_order + 1);

        for (auto& e : entries)
            groups[e.order].push_back(&e);

        for (auto& g : groups)
            std::sort(g.begin(), g.end(),
                    [](Entry* a, Entry* b)
                    {   
                        // return a->mi < b->mi;

                        if (!NumericUtils::doubleNearlyEqual(a->mi, b->mi))
                            return a->mi < b->mi;

                        // tie-breaker: deterministic
                        return a->mask < b->mask;
                    });

        const double eps = 1e-12;

        auto get_max = [&](size_t m)
        {
            if (m >= groups.size() || groups[m].empty())
                return 0.0;

            return groups[m].back()->mi;
        };

        /***********************************************************
        * Order 1 decomposition
        ***********************************************************/
        if (!groups[1].empty())
        {
            double prev = 0.0;

            for (size_t i = 0; i < groups[1].size(); ++i)
            {
                auto* e = groups[1][i];

                double delta = e->mi - prev;

                if (delta > eps)
                {
                    result.values.push_back(delta);

                    if (i == groups[1].size() - 1)
                        result.types.push_back(1);
                    else
                        result.types.push_back(0);

                    result.var_indices.push_back(mask_vars[e->mask]);
                }

                prev = e->mi;
            }
        }

        /***********************************************************
        * Higher order synergy
        ***********************************************************/
        for (size_t m = 2; m <= max_order; ++m)
        {
            if (groups[m].empty())
                continue;

            double max_prev = get_max(m - 1);

            for (size_t i = 0; i < groups[m].size(); ++i)
            {
                auto* e = groups[m][i];

                double prev = (i > 0) ? groups[m][i - 1]->mi : 0.0;

                double delta = 0.0;

                if (e->mi > max_prev + eps)
                {
                    if (prev >= max_prev)
                        delta = e->mi - prev;
                    else
                        delta = e->mi - max_prev;
                }

                if (delta > eps)
                {
                    result.values.push_back(delta);
                    result.types.push_back(2);
                    result.var_indices.push_back(mask_vars[e->mask]);
                }
            }
        }

        /***********************************************************
        * Optional normalization
        ***********************************************************/
        if (normalize)
        {
            double sum = 0.0;

            for (size_t i = 0; i < result.values.size(); ++i)
                if (result.types[i] != 3)
                    sum += result.values[i];

            if (sum > eps)
            {
                for (size_t i = 0; i < result.values.size(); ++i)
                    if (result.types[i] != 3)
                        result.values[i] /= sum;
            }
        }

        /***********************************************************
        * Information loss
        ***********************************************************/
        uint64_t all_sources = ((1ULL << n_vars) - 1) ^ 1ULL;

        uint64_t joint = all_sources | 1ULL;

        double ht = H[1];
        double hs = H[all_sources];
        double hts = H[joint];

        double leak = 0.0;

        if (!std::isnan(ht) && ht > eps)
        {
            double ce = hts - hs;
            leak = ce / ht;
            leak = std::max(0.0, std::min(1.0, leak));
        }

        result.values.push_back(leak);
        result.types.push_back(3);
        result.var_indices.push_back({});

        return result;
    }

}

#endif
