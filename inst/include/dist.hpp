/***************************************************************
 *  dist.hpp
 *
 *  High performance distance measurements
 *  for scalar, vector and matrix data.
 *
 *  Supported distance methods:
 *      "euclidean"  : sqrt(sum((x - y)^2))
 *      "maximum"    : max(|x - y|)
 *      "manhattan"  : sum(|x - y|)
 *
 *  NA handling:
 *      When na_rm = true:
 *          NaN values are removed pairwise before calculation.
 *          If all elements are removed, result is NaN.
 *
 *  Functions:
 *      dist(scalar, scalar)
 *      dist(vector, scalar)
 *      dist(vector, vector)
 *      dist(matrix)
 *
 *  Author: Wenbo Lyu (Github: @SpatLyu)
 *  License: GPL-3
 ***************************************************************/

#ifndef DIST_HPP
#define DIST_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <stdexcept>
#include <algorithm>

namespace Dist
{   
    /***********************************************************
     * Scalar - Scalar
     ***********************************************************/
    inline double dist(
        const double scalar1,
        const double scalar2,
        bool na_rm = true)
    {
        if (na_rm && (std::isnan(scalar1) || std::isnan(scalar2)))
            return std::numeric_limits<double>::quiet_NaN();

        return std::abs(scalar1 - scalar2);
    }

    /***********************************************************
     * Scalar - Vector
     * Result length equals vector length
     ***********************************************************/
    inline std::vector<double> dist(
        const double scalar,
        const std::vector<double>& vec,
        bool na_rm = true)
    {
        std::vector<double> result(vec.size(),
            std::numeric_limits<double>::quiet_NaN());

        for (size_t i = 0; i < vec.size(); ++i)
        {
            if (na_rm && (std::isnan(vec[i]) || std::isnan(scalar)))
                continue;

            result[i] = std::abs(vec[i] - scalar);
        }

        return result;
    }

    /***********************************************************
     * Vector - Scalar
     * Result length equals vector length
     ***********************************************************/
    inline double dist(
        const std::vector<double>& vec,
        const double scalar,
        std::string method = "euclidean",
        bool na_rm = true)
    {
        std::vector<double> result(vec.size(),
            std::numeric_limits<double>::quiet_NaN());

        for (size_t i = 0; i < vec.size(); ++i)
        {
            if (na_rm && (std::isnan(vec[i]) || std::isnan(scalar)))
                continue;

            result[i] = std::abs(vec[i] - scalar);
        }

        return result;
    }

    /***********************************************************
     * Vector - Vector
     * Element-wise distance
     ***********************************************************/
    inline std::vector<double> dist(
        const std::vector<double>& vec1,
        const std::vector<double>& vec2,
        std::string method = "euclidean",
        bool na_rm = true)
    {
        if (vec1.size() != vec2.size())
            throw std::invalid_argument("Vectors must have equal length.");

        std::vector<double> x_clean;
        std::vector<double> y_clean;

        if (!na_rm)
        {
            for (size_t i = 0; i < vec1.size(); ++i)
            {
                if (std::isnan(vec1[i]) || std::isnan(vec2[i]))
                    return std::vector<double>(
                        vec1.size(),
                        std::numeric_limits<double>::quiet_NaN());
            }

            std::vector<double> result(vec1.size());
            for (size_t i = 0; i < vec1.size(); ++i)
            {
                result[i] = std::abs(vec1[i] - vec2[i]);
            }
            return result;
        }

        std::vector<double> result(vec1.size(),
            std::numeric_limits<double>::quiet_NaN());

        for (size_t i = 0; i < vec1.size(); ++i)
        {
            if (!std::isnan(vec1[i]) && !std::isnan(vec2[i]))
            {
                result[i] = std::abs(vec1[i] - vec2[i]);
            }
        }

        return result;
    }

    /***********************************************************
     * Utilities
     ***********************************************************/
    inline bool is_na(double x)
    {
        return std::isnan(x);
    }

    inline double compute_distance(
        const std::vector<double>& x,
        const std::vector<double>& y,
        const std::string& method)
    {
        if (x.size() != y.size())
            throw std::invalid_argument("Vectors must have equal length.");

        if (x.empty())
            return std::numeric_limits<double>::quiet_NaN();

        if (method == "euclidean")
        {
            double sum = 0.0;
            for (size_t i = 0; i < x.size(); ++i)
            {
                double d = x[i] - y[i];
                sum += d * d;
            }
            return std::sqrt(sum);
        }
        else if (method == "manhattan")
        {
            double sum = 0.0;
            for (size_t i = 0; i < x.size(); ++i)
            {
                sum += std::abs(x[i] - y[i]);
            }
            return sum;
        }
        else if (method == "maximum")
        {
            double maxv = 0.0;
            for (size_t i = 0; i < x.size(); ++i)
            {
                maxv = std::max(maxv, std::abs(x[i] - y[i]));
            }
            return maxv;
        }
        else
        {
            throw std::invalid_argument("Unsupported distance method.");
        }
    }

    inline void remove_na_pairwise(
        const std::vector<double>& x,
        const std::vector<double>& y,
        std::vector<double>& x_clean,
        std::vector<double>& y_clean)
    {
        x_clean.clear();
        y_clean.clear();

        for (size_t i = 0; i < x.size(); ++i)
        {
            if (!is_na(x[i]) && !is_na(y[i]))
            {
                x_clean.push_back(x[i]);
                y_clean.push_back(y[i]);
            }
        }
    }

    /***********************************************************
     * Matrix
     * Row-wise distance matrix
     ***********************************************************/
    inline std::vector<std::vector<double>> dist(
        std::vector<std::vector<double>>& mat,
        std::string method = "euclidean",
        bool na_rm = true)
    {
        const size_t n = mat.size();

        std::vector<std::vector<double>> result(
            n,
            std::vector<double>(n,
                std::numeric_limits<double>::quiet_NaN()));

        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = i; j < n; ++j)
            {
                std::vector<double> x_clean;
                std::vector<double> y_clean;

                if (mat[i].size() != mat[j].size())
                    throw std::invalid_argument("Matrix rows must have equal length.");

                if (na_rm)
                {
                    remove_na_pairwise(mat[i], mat[j], x_clean, y_clean);

                    if (x_clean.empty())
                    {
                        result[i][j] = result[j][i] =
                            std::numeric_limits<double>::quiet_NaN();
                        continue;
                    }

                    result[i][j] = result[j][i] =
                        compute_distance(x_clean, y_clean, method);
                }
                else
                {
                    result[i][j] = result[j][i] =
                        compute_distance(mat[i], mat[j], method);
                }
            }
        }

        return result;
    }

} // namespace Dist

#endif // DIST_HPP
