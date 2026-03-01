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
                if (i == p) continue; // Skip self-matching

                double sum_s = 0.0;
                double maxv = 0.0;
                size_t n_valid = 0;

                for (size_t j = 0; j < embedding[p].size(); ++j) {
                    if (!std::isnan(embedding[i][j]) && !std::isnan(embedding[p][j])) {
                        double diff = embedding[i][j] - embedding[p][j];
                        if (method == "euclidean") {
                            sum_s += diff * diff;
                        }
                        else if (method == "manhattan") {
                            sum_s += std::abs(diff);
                        }
                        else if (method == "maximum") {
                            double ad = std::abs(diff);
                            if (ad > maxv) maxv = ad;
                        }
                        else {
                            throw std::invalid_argument("Unsupported distance method.");
                        }

                        ++n_valid;
                    }
                }

                if (n_valid > 0) {
                    if (method == "euclidean")
                        distances.push_back(std::sqrt(sum_s));
                    else if (method == "manhattan")
                        distances.push_back(sum_s);
                    else
                        distances.push_back(maxv);
                    
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

            pred[p] = prediction / total_weight;
        }

        return pred;

    }

}// namespace Projection

#endif // PROJECTION_HPP
