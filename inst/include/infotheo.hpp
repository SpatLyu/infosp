/***************************************************************
 *  infotheo.hpp
 *
 *  High performance information theoretic measurements
 *  for discrete pattern data.
 *
 *  Pattern encoding:
 *      uint8_t
 *      0 = NA
 *      1 = down
 *      2 = flat
 *      3 = up
 *
 *  Data layout:
 *      Pattern        = std::vector<uint8_t>
 *      PatternSeries  = std::vector<Pattern>        // one variable
 *      Matrix         = std::vector<PatternSeries>  // mat[var][obs]
 *
 *  Functions:
 *      Entropy
 *      JE   Joint Entropy
 *      CE   Conditional Entropy
 *      MI   Mutual Information
 *      CMI  Conditional Mutual Information
 *
 *  Author: Wenbo Lyu (Github: @SpatLyu)
 *  License: GPL-3
 ***************************************************************/

#ifndef INFOTHEO_HPP
#define INFOTHEO_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace InfoTheo
{
    using Pattern       = std::vector<uint8_t>;
    using PatternSeries = std::vector<Pattern>;
    using Matrix        = std::vector<PatternSeries>;

    /***********************************************************
     * Utilities
     ***********************************************************/
    inline double convert_log_base(double x, double base)
    {
        if (x <= 0.0) return 0.0;
        if (!(base > 0.0) || std::abs(base - 1.0) < 1e-12)
        {
          throw std::invalid_argument("Log base must be positive and not equal to 1.");
        }
        return x / std::log(base);
    }

    /***********************************************************
     * Strong 64bit mix for hash
     ***********************************************************/
    static inline uint64_t mix64(uint64_t x)
    {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    /***********************************************************
     * PackedKey
     * Each symbol uses 2 bits.
     ***********************************************************/
    struct PackedKey
    {
        std::vector<uint64_t> blocks;

        bool operator==(const PackedKey& other) const noexcept
        {
            return blocks == other.blocks;
        }
    };

    struct PackedKeyHash
    {
        size_t operator()(const PackedKey& k) const noexcept
        {
            uint64_t h = 0;
            for (uint64_t b : k.blocks)
            {
                h ^= mix64(b + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
            }
            return static_cast<size_t>(h);
        }
    };

    /***********************************************************
     * Pack one pattern sequence into PackedKey
     * Return false if NA detected and NA_rm == true
     ***********************************************************/
    inline bool pack_pattern(
        const Pattern& p,
        PackedKey& key,
        bool NA_rm)
    {
        key.blocks.clear();

        uint64_t cur = 0;
        int shift = 0;

        for (uint8_t x : p)
        {
            if (NA_rm && x == 0)
                return false;

            uint64_t val = static_cast<uint64_t>(x & 0x3);
            cur |= (val << shift);
            shift += 2;

            if (shift >= 64)
            {
                key.blocks.push_back(cur);
                cur = 0;
                shift = 0;
            }
        }

        if (shift > 0)
            key.blocks.push_back(cur);

        return true;
    }

    /***********************************************************
     * Pack multiple variables for one observation
     ***********************************************************/
    inline bool pack_observation(
        const Matrix& mat,
        const std::vector<size_t>& vars,
        size_t obs_id,
        PackedKey& key,
        bool NA_rm = false)
    {
        key.blocks.clear();

        uint64_t cur = 0;
        int shift = 0;

        for (size_t v : vars)
        {
            const Pattern& p = mat[v][obs_id];

            for (uint8_t x : p)
            {
                if (NA_rm && x == 0)
                    return false;

                uint64_t val = static_cast<uint64_t>(x & 0x3);
                cur |= (val << shift);
                shift += 2;

                if (shift >= 64)
                {
                    key.blocks.push_back(cur);
                    cur = 0;
                    shift = 0;
                }
            }
        }

        if (shift > 0)
            key.blocks.push_back(cur);

        return true;
    }

    /***********************************************************
     * Entropy of one variable
     ***********************************************************/
    inline double Entropy(
        const PatternSeries& series,
        double base = 2.0,
        bool NA_rm = false)
    {
        if (series.empty())
            return std::numeric_limits<double>::quiet_NaN();

        std::unordered_map<PackedKey, size_t, PackedKeyHash> freq;
        freq.reserve(series.size() * 1.3);

        PackedKey key;
        size_t n_valid = 0;

        for (const auto& p : series)
        {
            if (!pack_pattern(p, key, NA_rm))
                continue;

            ++freq[key];
            ++n_valid;
        }

        if (n_valid == 0) return std::numeric_limits<double>::quiet_NaN();

        double h = 0.0;
        for (const auto& kv : freq)
        {
            double p = static_cast<double>(kv.second) / n_valid;
            h -= p * std::log(p);
        }
        return convert_log_base(h,base);
    }

    /***********************************************************
     * Joint Entropy
     ***********************************************************/
    inline double JE(
        const Matrix& mat,
        const std::vector<size_t>& vars,
        double base = 2.0,
        bool NA_rm = false)
    {
        if (mat.empty() || vars.empty())
            return std::numeric_limits<double>::quiet_NaN();

        const size_t n_obs = mat[0].size();
        const size_t n_cols = mat.size();

        std::unordered_set<size_t> valid_vars;
        valid_vars.reserve(vars.size());
        for (size_t idx : vars) {
          if (idx < n_cols) {
            valid_vars.insert(idx);
          }
        }
        std::vector<size_t> clean_vars(valid_vars.begin(), valid_vars.end());

        std::unordered_map<PackedKey, size_t, PackedKeyHash> freq;
        freq.reserve(n_obs * 1.3);

        PackedKey key;
        size_t n_valid = 0;

        for (size_t i = 0; i < n_obs; ++i)
        {
            if (!pack_observation(mat, clean_vars, i, key, NA_rm))
                continue;

            ++freq[key];
            ++n_valid;
        }

        if (n_valid == 0)
            return std::numeric_limits<double>::quiet_NaN();

        double h = 0.0;
        for (const auto& kv : freq)
        {
            double p = static_cast<double>(kv.second) / n_valid;
            h -= p * std::log(p);
        }
        return convert_log_base(h,base);
    }

    /***********************************************************
     * Conditional Entropy H(target | conds)
     ***********************************************************/
    inline double CE(
        const Matrix& mat,
        const std::vector<size_t>& target,
        const std::vector<size_t>& conds,
        double base = 2.0,
        bool NA_rm = false)
    {
        if (mat.empty() || target.empty()) return std::numeric_limits<double>::quiet_NaN();

        if (conds.empty())
        {
          throw std::invalid_argument("The conds parameter can not be empty.");
        }

        std::vector<size_t> tc = conds;
        tc.insert(tc.end(), target.begin(), target.end());

        return JE(mat, tc, base, NA_rm)
             - JE(mat, conds, base, NA_rm);
    }

    /***********************************************************
     * Mutual Information I(target ; interact)
     ***********************************************************/
    inline double MI(
        const Matrix& mat,
        const std::vector<size_t>& target,
        const std::vector<size_t>& interact,
        double base = 2.0,
        bool NA_rm = false)
    {
        if (mat.empty() || target.empty()) return std::numeric_limits<double>::quiet_NaN();

        if (interact.empty())
        {
          throw std::invalid_argument("The interact parameter can not be empty.");
        }

        std::vector<size_t> ti = interact;
        ti.insert(ti.end(), target.begin(), target.end());

        return JE(mat, target, base, NA_rm) +
               JE(mat, interact, base, NA_rm) -
               JE(mat, ti, base, NA_rm);
    }

    /***********************************************************
     * Conditional Mutual Information
     * I(target ; interact | conds)
     ***********************************************************/
    inline double CMI(
        const Matrix& mat,
        const std::vector<size_t>& target,
        const std::vector<size_t>& interact,
        const std::vector<size_t>& conds,
        double base = 2.0,
        bool NA_rm = false)
    {
        if (mat.empty() || target.empty()) return std::numeric_limits<double>::quiet_NaN();

        if (interact.empty() || conds.empty())
        {
          throw std::invalid_argument("The interact / conds parameter can not be empty.");
        }

        std::vector<size_t> ct  = conds;
        std::vector<size_t> ci  = conds;
        std::vector<size_t> cti = conds;

        ct.insert(ct.end(), target.begin(), target.end());
        ci.insert(ci.end(), interact.begin(), interact.end());
        cti.insert(cti.end(), target.begin(), target.end());
        cti.insert(cti.end(), interact.begin(), interact.end());

        return JE(mat, ct,  base, NA_rm)
             + JE(mat, ci,  base, NA_rm)
             - JE(mat, conds, base, NA_rm)
             - JE(mat, cti, base, NA_rm);
    }

} // namespace InfoTheo

#endif // INFOTHEO_HPP
