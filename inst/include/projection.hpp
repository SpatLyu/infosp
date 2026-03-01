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

namespace Projection
{
    inline std::vector<double> Simplex(
        const std::vector<std::vector<double>>& embedding,
        const std::vector<double>& target,
        const std::vector<size_t>& lib,
        const std::vector<size_t>& pred,
        size_t num_neighbors = 4,
        std::string method = "euclidean")
    {
        size_t N = target.size();
        std::vector<double> pred(N, std::numeric_limits<double>::quiet_NaN());

        if (num_neighbors == 0 || num_neighbors >= N) {
            return pred;  // no valid neighbors to use, return all NaNs
        }

        for (size_t pi = 0; pi < pred_indices.size(); ++pi) {
            int p = pred_indices[pi];

            // // Skip if target at prediction index is NaN
            // if (std::isnan(target[p])) {
            //   continue;
            // }

            // Compute distances only for valid vector pairs (exclude NaNs)
            std::vector<double> distances;
            distances.reserve(lib_indices.size());
            // keep track of libs corresponding to valid distances
            std::vector<int> valid_libs;
            valid_libs.reserve(lib_indices.size());

            for (int i : lib_indices) {
            if (i == p) continue; // Skip self-matching

            double sum_sq = 0.0;
            std::size_t count = 0;
            for (size_t j = 0; j < vectors[p].size(); ++j) {
                if (!std::isnan(vectors[i][j]) && !std::isnan(vectors[p][j])) {
                double diff = vectors[i][j] - vectors[p][j];
                // sum_sq += (dist_metric == 1) ? std::abs(diff) : diff * diff;
                if (dist_metric == 1) {
                    sum_sq += std::abs(diff); // L1
                } else {
                    sum_sq += diff * diff;    // L2
                }
                ++count;
                }
            }
            if (count > 0) {
                if (dist_metric == 1) {  // L1
                distances.push_back(sum_sq / (dist_average ? static_cast<double>(count) : 1.0));
                } else {                 // L2
                distances.push_back(std::sqrt(sum_sq / (dist_average ? static_cast<double>(count) : 1.0)));
                }
                valid_libs.push_back(i);
            }
            }

            // If no valid distances found, prediction is NaN
            if (distances.empty()) {
            continue;
            }

            // Adjust number of neighbors to available valid libs
            size_t k = std::min(static_cast<size_t>(num_neighbors), distances.size());

            // Prepare indices for sorting
            std::vector<size_t> neighbors(distances.size());
            std::iota(neighbors.begin(), neighbors.end(), 0);

            // Partial sort to find k nearest neighbors by distance
            std::partial_sort(
            neighbors.begin(), neighbors.begin() + k, neighbors.end(),
            [&](size_t a, size_t b) {
                if (!doubleNearlyEqual(distances[a], distances[b])) {
                return distances[a] < distances[b];
                } else {
                return a < b;
                }
            });

            double min_distance = distances[neighbors[0]];

            // Compute weights for neighbors
            std::vector<double> weights(k);
            if (doubleNearlyEqual(min_distance,0.0)) { // Perfect match found
            std::fill(weights.begin(), weights.end(), 0.000001);
            for (size_t i = 0; i < k; ++i) {
                if (doubleNearlyEqual(distances[neighbors[i]],0.0)) {
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

            pred[p] = prediction / total_weight;
        }

        return pred;

    }

}// namespace Projection

#endif // PROJECTION_HPP
