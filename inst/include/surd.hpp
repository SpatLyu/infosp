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
 * Vector Hash
 ***********************************************************/
struct VecHash
{
    size_t operator()(const std::vector<size_t>& v) const noexcept
    {
        size_t h = v.size();
        constexpr size_t golden = 0x9e3779b97f4a7c15ULL;

        for (size_t x : v)
        {
            h ^= std::hash<size_t>{}(x) + golden + (h << 6) + (h >> 2);
        }

        return h;
    }
};

/***********************************************************
 * SURD Result Structure
 ***********************************************************/
struct SURDRes
{
    std::vector<double> values;
    std::vector<uint8_t> types;
    std::vector<std::vector<size_t>> var_indices;

    size_t size() const noexcept { return values.size(); }
};

/***********************************************************
 * Entropy Cache
 ***********************************************************/
using EntropyMap =
std::unordered_map<std::vector<size_t>, double, VecHash>;

/***********************************************************
 * Parallel entropy task
 ***********************************************************/
inline void entropy_task(
    const Matrix& mat,
    const std::vector<std::vector<size_t>>& vars,
    EntropyMap& local_cache,
    double base,
    bool na_rm)
{
    for (auto v : vars)
    {
        std::sort(v.begin(), v.end());

        double h = InfoTheo::JE(mat, v, base, na_rm);

        local_cache.emplace(std::move(v), h);
    }
}

/***********************************************************
 * Parallel entropy precomputation
 ***********************************************************/
inline EntropyMap precompute_entropies(
    const Matrix& mat,
    const std::vector<std::vector<size_t>>& subsets,
    double base,
    bool na_rm,
    size_t n_threads)
{
    size_t hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    if (n_threads == 0 || n_threads > hw)
        n_threads = hw;

    std::vector<std::vector<size_t>> tasks;
    tasks.reserve(subsets.size() * 2 + 1);

    for (auto s : subsets)
    {
        std::sort(s.begin(), s.end());
        tasks.push_back(s);

        s.push_back(0);
        std::sort(s.begin(), s.end());
        tasks.push_back(s);
    }

    tasks.push_back({0});

    size_t total = tasks.size();
    size_t chunk = (total + n_threads - 1) / n_threads;

    std::vector<std::thread> threads;
    std::vector<EntropyMap> local_maps(n_threads);

    for (size_t t = 0; t < n_threads; ++t)
    {
        size_t start = t * chunk;
        size_t end = std::min(start + chunk, total);

        if (start >= end)
            break;

        std::vector<std::vector<size_t>> sub(
            tasks.begin() + start,
            tasks.begin() + end
        );

        threads.emplace_back(
            entropy_task,
            std::cref(mat),
            std::move(sub),
            std::ref(local_maps[t]),
            base,
            na_rm
        );
    }

    for (auto& th : threads)
        th.join();

    EntropyMap cache;
    cache.reserve(tasks.size());

    for (auto& m : local_maps)
    {
        for (auto& kv : m)
            cache.emplace(std::move(kv));
    }

    return cache;
}

/***********************************************************
 * Safe entropy lookup
 ***********************************************************/
inline double get_entropy(
    const EntropyMap& cache,
    const std::vector<size_t>& key)
{
    auto it = cache.find(key);

    if (it == cache.end())
        return NAN;

    return it->second;
}

/***********************************************************
 * Compute MI using entropy cache
 ***********************************************************/
inline double compute_mi(
    const EntropyMap& cache,
    std::vector<size_t> subset)
{
    std::sort(subset.begin(), subset.end());

    std::vector<size_t> ts = subset;
    ts.push_back(0);
    std::sort(ts.begin(), ts.end());

    double ht = get_entropy(cache, {0});
    double hs = get_entropy(cache, subset);
    double hts = get_entropy(cache, ts);

    if (std::isnan(ht) || std::isnan(hs) || std::isnan(hts))
        return NAN;

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
    size_t threads = 1)
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

    auto cache =
        precompute_entropies(mat, subsets, base, na_rm, threads);

    struct Entry
    {
        double mi;
        std::vector<size_t> vars;
        size_t order;
    };

    std::vector<Entry> entries;
    entries.reserve(subsets.size());

    for (auto s : subsets)
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
                  [](Entry* a, Entry* b)
                  { return a->mi < b->mi; });

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
                    result.types.push_back(1);   // unique
                else
                    result.types.push_back(0);   // redundant

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
                result.types.push_back(2); // synergy
                result.var_indices.push_back(g[i]->vars);
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
    std::vector<size_t> all_sources = source_vars;

    std::vector<size_t> ts = all_sources;
    ts.push_back(0);

    std::sort(ts.begin(), ts.end());

    double h_target = get_entropy(cache, {0});
    double h_sources = get_entropy(cache, all_sources);
    double h_all = get_entropy(cache, ts);

    double leak = 0.0;

    if (h_target > eps)
    {
        double ce = h_all - h_sources;
        leak = ce / h_target;
        leak = std::max(0.0, std::min(1.0, leak));
    }

    result.values.push_back(leak);
    result.types.push_back(3);
    result.var_indices.push_back({});

    return result;
}

} // namespace SURD

#endif
