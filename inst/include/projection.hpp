/***************************************************************
 *  projection.hpp
 *
 *  High performance state space projection methods
 *  for nonlinear state prediction.
 *
 *  Core method:
 *      Simplex Projection
 *
 *  Description:
 *      Perform nearest neighbor prediction in reconstructed
 *      state space using weighted simplex projection.
 *
 *  Distance methods:
 *      "euclidean"  : sqrt(sum((x - y)^2))
 *      "maximum"    : max(|x - y|)
 *      "manhattan"  : sum(|x - y|)
 *
 *  NA handling:
 *      NaN values inside embedding vectors are ignored
 *      pairwise when computing distances.
 *      If no valid dimension exists, the distance is skipped.
 *
 *  Data layout:
 *      embedding  : std::vector<std::vector<double>>
 *                   embedding[row][dimension]
 *
 *      target     : std::vector<double>
 *                   observed values corresponding to rows
 *
 *      lib        : library indices used as neighbors
 *      pred       : prediction indices
 *
 *  Output:
 *      A vector of predictions aligned to target length.
 *      Non predicted positions are NaN.
 *
 *  Author: Wenbo Lyu (Github: @SpatLyu)
 *  License: GPL-3
 ***************************************************************/

#ifndef PROJECTION_HPP
#define PROJECTION_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <utility>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include "numericutils.hpp"
#include "distance.hpp"

namespace Projection
{   
    /***********************************************************
     * Simplex Projection
     *
     * Parameters:
     *      embedding     reconstructed state space
     *      target        observed response vector
     *      lib           library row indices
     *      pred          prediction row indices
     *      num_neighbors number of nearest neighbors
     *      method        distance metric
     *
     * Returns:
     *      prediction vector aligned to target length
     *
     * Algorithm:
     *      1. For each prediction index p
     *      2. Compute distances between embedding[p]
     *         and embedding[lib[i]]
     *      3. Select k nearest neighbors
     *      4. Compute exponential weights
     *      5. Produce weighted average prediction
     *
     * Notes:
     *      Self matching is excluded.
     *      If no valid neighbor exists, prediction is NaN.
     ***********************************************************/
    inline std::vector<double> Simplex(
        const std::vector<std::vector<double>>& embedding,
        const std::vector<double>& target,
        const std::vector<size_t>& lib,
        const std::vector<size_t>& pred,
        size_t num_neighbors = 4,
        std::string method = "euclidean")
    {   
        const Dist::DistanceMethod dist_method = Dist::parseDistanceMethod(method);
        if (dist_method == Dist::DistanceMethod::Invalid) {
            throw std::invalid_argument("Unsupported distance method: " + method);
        }

        size_t N = target.size();
        std::vector<double> pred_t(N, std::numeric_limits<double>::quiet_NaN());

        if (num_neighbors == 0 || num_neighbors > lib.size()) {
            return pred_t;  // no valid neighbors to use, return all NaNs
        }

        for (size_t pi = 0; pi < pred.size(); ++pi) {
            size_t p = pred[pi];

            // // Skip if target at prediction index is NaN
            // if (std::isnan(target[p])) {
            //   continue;
            // }

            // Compute distances only for valid vector pairs (exclude NaNs)
            std::vector<double> distances;
            distances.reserve(lib.size());
            // keep track of libs corresponding to valid distances
            std::vector<size_t> valid_libs;
            valid_libs.reserve(lib.size());

            for (size_t i : lib) {
                // Skip self-matching
                if (i == p) continue; 
                
                // Skip non-valid library
                if (std::isnan(target[i])) continue; 

                double sum_s = 0.0;
                double maxv = 0.0;
                size_t n_valid = 0;

                for (size_t j = 0; j < embedding[p].size(); ++j) {
                    if (!std::isnan(embedding[i][j]) && !std::isnan(embedding[p][j])) {
                        double diff = embedding[i][j] - embedding[p][j];
                        // if (method == "euclidean") {
                        //     sum_s += diff * diff;
                        // }
                        // else if (method == "manhattan") {
                        //     sum_s += std::abs(diff);
                        // }
                        // else if (method == "maximum") {
                        //     double ad = std::abs(diff);
                        //     if (ad > maxv) maxv = ad;
                        // }
                        // else {
                        //     throw std::invalid_argument("Unsupported distance method.");
                        // }

                        switch (dist_method) {
                            case Dist::DistanceMethod::Euclidean:
                                sum_s += diff * diff;
                                break;
                            case Dist::DistanceMethod::Manhattan:
                                sum_s += std::abs(diff);
                                break;
                            case Dist::DistanceMethod::Maximum:
                                {
                                    double ad = std::abs(diff);
                                    if (ad > maxv) maxv = ad;
                                }
                                break;
                            default:
                                break; 
                        }

                        ++n_valid;
                    }
                }

                if (n_valid > 0) {
                    // if (method == "euclidean")
                    //     distances.push_back(std::sqrt(sum_s));
                    // else if (method == "manhattan")
                    //     distances.push_back(sum_s);
                    // else
                    //     distances.push_back(maxv);

                    if (dist_method == Dist::DistanceMethod::Euclidean)
                        distances.push_back(std::sqrt(sum_s));
                    else if (dist_method == Dist::DistanceMethod::Manhattan)
                        distances.push_back(sum_s);
                    else
                        distances.push_back(maxv);  // maximum
                    
                    valid_libs.push_back(i);    
                }
            }

            // If no valid distances found, prediction is NaN
            if (distances.empty()) {
                continue;
            }

            // Adjust number of neighbors to available valid libs
            const size_t k = std::min(num_neighbors, distances.size());

            // Prepare indices for sorting
            std::vector<size_t> neighbors(distances.size());
            std::iota(neighbors.begin(), neighbors.end(), 0);

            // Partial sort to find k nearest neighbors by distance
            std::partial_sort(
                neighbors.begin(), neighbors.begin() + k, neighbors.end(),
                [&](size_t a, size_t b) {
                    if (!NumericUtils::doubleNearlyEqual(distances[a], distances[b])) {
                        return distances[a] < distances[b];
                    } else {
                        return a < b;
                    }
                }
            );

            double min_distance = distances[neighbors[0]];

            // Compute weights for neighbors
            std::vector<double> weights(k);
            if (NumericUtils::doubleNearlyEqual(min_distance,0.0)) { // Perfect match found
                std::fill(weights.begin(), weights.end(), 0.000001);
                for (size_t i = 0; i < k; ++i) {
                    if (NumericUtils::doubleNearlyEqual(distances[neighbors[i]],0.0)) {
                        weights[i] = 1.0;
                    }
                }
            } else {
                for (size_t i = 0; i < k; ++i) {
                    weights[i] = std::exp(-distances[neighbors[i]] / min_distance);
                    if (weights[i] < 0.000001) {
                        weights[i] = 0.000001;
                    }
                }
            }

            double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);

            // Calculate weighted prediction, ignoring any NaN targets
            // (No NaNs here, as NaN values in the corresponding components of lib and pred are excluded in advance.)
            double prediction = 0.0;
            for (size_t i = 0; i < k; ++i) {
                prediction += weights[i] * target[valid_libs[neighbors[i]]];
            }

            pred_t[p] = prediction / total_weight;
        }

        return pred_t;
    }

} // namespace Projection

#endif // PROJECTION_HPP
