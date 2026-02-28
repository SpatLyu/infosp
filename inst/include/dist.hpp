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
        const double scalar2)
    {
        if (std::isnan(scalar1) || std::isnan(scalar2))
            return std::numeric_limits<double>::quiet_NaN();

        return std::abs(scalar1 - scalar2);
    }

    /***********************************************************
     * Scalar - Vector
     * Result length equals vector length
     ***********************************************************/
    inline std::vector<double> dist(
        const double scalar,
        const std::vector<double>& vec)
    {
        std::vector<double> result(vec.size(),
            std::numeric_limits<double>::quiet_NaN());

        if (std::isnan(scalar)) return result;

        for (size_t i = 0; i < vec.size(); ++i)
        {
            if (!std::isnan(vec[i]))
            {
                result[i] = std::abs(vec[i] - scalar);
            }
        }

        return result;
    }

    inline std::vector<double> dist(
        const std::vector<double>& vec,
        const double scalar)
    {
        return dist(scalar, vec);
    }

    /***********************************************************
    * Vector - Scalar
    * Scalar is internally expanded to vector length
    * Result is a single double distance value
    ***********************************************************/
    inline double dist(
        const std::vector<double>& vec,
        const double scalar,
        std::string method = "euclidean",
        bool na_rm = true)
    {
        if (vec.empty() || std::isnan(scalar))
            return std::numeric_limits<double>::quiet_NaN();

        double sum = 0.0;
        double maxv = 0.0;
        size_t n_valid = 0;

        for (size_t i = 0; i < vec.size(); ++i)
        {
            if (na_rm && std::isnan(vec[i]))
                continue;

            if (!na_rm && std::isnan(vec[i]))
                return std::numeric_limits<double>::quiet_NaN();

            double diff = vec[i] - scalar;
            double ad   = std::abs(diff);

            if (method == "euclidean")
            {
                sum += diff * diff;
            }
            else if (method == "manhattan")
            {
                sum += ad;
            }
            else if (method == "maximum")
            {
                if (ad > maxv) maxv = ad;
            }
            else
            {
                throw std::invalid_argument("Unsupported distance method.");
            }

            ++n_valid;
        }

        if (n_valid == 0)
            return std::numeric_limits<double>::quiet_NaN();

        if (method == "euclidean")
            return std::sqrt(sum);
        else if (method == "manhattan")
            return sum;
        else
            return maxv;  // maximum
    }

    inline double dist(
        const double scalar,
        const std::vector<double>& vec,
        std::string method = "euclidean",
        bool na_rm = true)
    {
        return dist(vec, scalar, method, na_rm);
    }

    /***********************************************************
     * Vector - Vector
     * Element-wise distance
     ***********************************************************/
    inline std::vector<double> dist(
        const std::vector<double>& vec1,
        const std::vector<double>& vec2)
    {
        if (vec1.size() != vec2.size())
            throw std::invalid_argument("Vectors must have equal length.");

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
     * Vector - Vector
     * Result is a single double distance value
     ***********************************************************/
    inline double dist(
        const std::vector<double>& vec1,
        const std::vector<double>& vec2,
        std::string method = "euclidean",
        bool na_rm = true)
    {   
        if (vec1.empty() || vec1.empty() || vec1.size() != vec2.size())
            return std::numeric_limits<double>::quiet_NaN();

        double sum = 0.0;
        double maxv = 0.0;
        size_t n_valid = 0;

        for (size_t i = 0; i < vec1.size(); ++i)
        {   
            bool element_has_na = std::isnan(vec1[i]) || std::isnan(vec2[i]);

            if (element_has_na && na_rm)
                continue;

            if (element_has_na && !na_rm)
                return std::numeric_limits<double>::quiet_NaN();

            double diff = vec1[i] - vec2[i];
            double ad   = std::abs(diff);

            if (method == "euclidean")
            {
                sum += diff * diff;
            }
            else if (method == "manhattan")
            {
                sum += ad;
            }
            else if (method == "maximum")
            {
                if (ad > maxv) maxv = ad;
            }
            else
            {
                throw std::invalid_argument("Unsupported distance method.");
            }

            ++n_valid;
        }

        if (n_valid == 0)
            return std::numeric_limits<double>::quiet_NaN();

        if (method == "euclidean")
            return std::sqrt(sum);
        else if (method == "manhattan")
            return sum;
        else
            return maxv;  // maximum
    }

    /***********************************************************
     * Matrix
     * Row-wise distance matrix
     ***********************************************************/
    inline std::vector<std::vector<double>> dist(
        const std::vector<std::vector<double>>& mat,
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
