/**********************************************************************
 * File: surd.hpp
 *
 * Synergistic-Unique-Redundant Decomposition (SURD)
 * for discrete pattern data.
 *
 * Implementation strategy:
 *
 *   1. Enumerate all non-empty subsets of source variables
 *   2. Precompute all required joint entropies
 *   3. Cache entropy values
 *   4. Compute mutual information using entropy identities
 *   5. Apply SURD increment decomposition
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
#include <map>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cmath>
#include "combn.hpp"
#include "infotheo.hpp"

namespace SURD
{

using Matrix = InfoTheo::Matrix;

/***********************************************************
 * SURD Result Structure
 ***********************************************************/
struct SURDRes
{
    std::vector<double> values;
    std::vector<uint8_t> types;
    std::vector<std::vector<size_t>> var_indices;

    size_t size() const { return values.size(); }
};

/***********************************************************
 * Entropy Cache
 ***********************************************************/
struct EntropyCache
{
    std::unordered_map<std::vector<size_t>, double> cache;
    std::mutex mutex;
};

/***********************************************************
 * Compute entropy for a batch of variable sets
 ***********************************************************/
inline void entropy_task(
    const Matrix& mat,
    const std::vector<std::vector<size_t>>& vars,
    EntropyCache& cache,
    double base,
    bool na_rm)
{
    for (const auto& v : vars)
    {
        double h = InfoTheo::JE(mat, v, base, na_rm);

        std::lock_guard<std::mutex> lock(cache.mutex);
        cache.cache[v] = h;
    }
}

/***********************************************************
 * Parallel entropy precomputation
 ***********************************************************/
inline EntropyCache precompute_entropies(
    const Matrix& mat,
    const std::vector<std::vector<size_t>>& subsets,
    double base,
    bool na_rm,
    size_t n_threads)
{
    EntropyCache cache;

    std::vector<std::vector<size_t>> tasks;
    tasks.reserve(subsets.size() * 2 + 1);

    for (const auto& s : subsets)
    {
        tasks.push_back(s);

        std::vector<size_t> ts = s;
        ts.push_back(0);
        tasks.push_back(ts);
    }

    tasks.push_back({0});

    size_t total = tasks.size();
    size_t chunk = (total + n_threads - 1) / n_threads;

    std::vector<std::thread> threads;

    for (size_t t = 0; t < n_threads; ++t)
    {
        size_t start = t * chunk;
        size_t end   = std::min(start + chunk, total);

        if (start >= end)
            break;

        std::vector<std::vector<size_t>> sub(tasks.begin()+start, tasks.begin()+end);

        threads.emplace_back(
            entropy_task,
            std::cref(mat),
            std::move(sub),
            std::ref(cache),
            base,
            na_rm
        );
    }

    for (auto& th : threads)
        th.join();

    return cache;
}

/***********************************************************
 * Compute MI using entropy cache
 ***********************************************************/
inline double compute_mi(
    const EntropyCache& cache,
    const std::vector<size_t>& subset)
{
    std::vector<size_t> ts = subset;
    ts.push_back(0);

    double ht  = cache.cache.at({0});
    double hs  = cache.cache.at(subset);
    double hts = cache.cache.at(ts);

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
    size_t n_threads = 1)
{
    SURDRes result;

    if (mat.size() < 2)
        return result;

    const size_t n_sources = mat.size() - 1;

    std::vector<size_t> source_vars(n_sources);
    for (size_t i = 0; i < n_sources; ++i)
        source_vars[i] = i + 1;

    auto subsets = Combn::GenSubsets(source_vars);

    if (subsets.empty())
        return result;

    EntropyCache cache =
        precompute_entropies(mat, subsets, base, na_rm, n_threads);

    struct Entry
    {
        double mi;
        std::vector<size_t> vars;
        size_t order;
    };

    std::vector<Entry> entries;
    entries.reserve(subsets.size());

    for (const auto& s : subsets)
    {
        double mi = compute_mi(cache, s);

        if (!std::isnan(mi))
        {
            mi = std::max(0.0, mi);
            entries.push_back({mi, s, s.size()});
        }
    }

    std::map<size_t, std::vector<Entry*>> groups;

    for (auto& e : entries)
        groups[e.order].push_back(&e);

    for (auto& [k,v] : groups)
        std::sort(v.begin(), v.end(),
                  [](Entry* a, Entry* b){ return a->mi < b->mi; });

    const double eps = 1e-12;

    auto get_max = [&](size_t m)
    {
        if (!groups.count(m)) return 0.0;
        return groups[m].back()->mi;
    };

    /***********************************************************
     * Order 1 decomposition
     ***********************************************************/
    if (groups.count(1))
    {
        auto& g = groups[1];
        double prev = 0.0;

        for (size_t i = 0; i < g.size(); ++i)
        {
            double delta = g[i]->mi - prev;

            if (delta > eps)
            {
                result.values.push_back(delta);

                if (i == g.size() - 1)
                    result.types.push_back(1);   // Unique
                else
                    result.types.push_back(0);   // Redundant

                result.var_indices.push_back(g[i]->vars);
            }

            prev = g[i]->mi;
        }
    }

    /***********************************************************
     * Higher-order synergy
     ***********************************************************/
    for (size_t m = 2; m <= n_sources; ++m)
    {
        if (!groups.count(m)) continue;

        double max_prev = get_max(m-1);
        auto& g = groups[m];

        for (size_t i = 0; i < g.size(); ++i)
        {
            double prev = (i > 0) ? g[i-1]->mi : 0.0;

            double delta = 0.0;

            if (g[i]->mi > max_prev + eps)
            {
                if (prev >= max_prev)
                    delta = g[i]->mi - prev;
                else
                    delta = g[i]->mi - max_prev;
            }

            if (delta > eps)
            {
                result.values.push_back(delta);
                result.types.push_back(2);   // Synergistic
                result.var_indices.push_back(g[i]->vars);
            }
        }
    }

    /***********************************************************
     * Optional normalization of decomposed components
     ***********************************************************/
    if (normalize)
    {
        double sum = 0.0;

        for (size_t i = 0; i < result.values.size(); ++i)
        {
            if (result.types[i] != 3)
                sum += result.values[i];
        }

        if (sum > eps)
        {
            for (size_t i = 0; i < result.values.size(); ++i)
            {
                if (result.types[i] != 3)
                    result.values[i] /= sum;
            }
        }
    }

    /***********************************************************
     * Information leak
     ***********************************************************/
    std::vector<size_t> all_sources = source_vars;

    double h_target = cache.cache.at({0});
    double h_sources = cache.cache.at(all_sources);

    std::vector<size_t> ts = all_sources;
    ts.push_back(0);

    double h_all = cache.cache.at(ts);

    double ce = h_all - h_sources;

    double leak = ce / h_target;
    leak = std::max(0.0, std::min(1.0, leak));

    result.values.push_back(leak);
    result.types.push_back(3);
    result.var_indices.push_back({});

    return result;
}

} // namespace SURD

#endif
