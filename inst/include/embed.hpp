/***************************************************************
 *
 *  File: embed.hpp
 *
 *  Spatial Embedding Utilities for Lattice and Grid Structures.
 *
 *  This header provides high performance utilities for generating
 *  spatial embeddings on:
 *
 *    - Arbitrary lattices defined by neighbor lists
 *    - Regular 2D grids using Moore neighborhoods
 *
 *  Core capabilities:
 *
 *    - Multi lag neighbor expansion on graphs
 *    - Lagged value aggregation
 *    - Grid index mapping utilities
 *    - Spatial embedding generation
 *
 *  Design principles:
 *
 *    - Type safety using std::size_t indices
 *    - No sentinel or magic values
 *    - Cache friendly memory layouts
 *    - Header only implementation
 *    - Robust NaN propagation
 *
 *  Author: Wenbo Lv
 *  License: GPL-3
 *
 ***************************************************************/

#ifndef EMBED_HPP
#define EMBED_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include <cstddef>
#include <utility>

namespace Embed {

/* =============================================================
 * Type aliases
 * ============================================================= */

using Index        = std::size_t;
using NeighborList = std::vector<Index>;
using NeighborMat  = std::vector<NeighborList>;
using Vector       = std::vector<double>;
using Matrix       = std::vector<Vector>;

/* =============================================================
 * ------------------- LATTICE OPERATORS -----------------------
 * ============================================================= */

/**
 * @brief Expand neighbors on a lattice up to a given lag.
 *
 * For lag = 0, each node returns itself.
 * For lag > 0, neighbors are recursively expanded and merged.
 *
 * No sentinel values are used. Empty neighbor sets remain empty.
 */
inline NeighborMat LaggedNeighbors(
    const NeighborMat& nb,
    size_t lag
) {
    const size_t n = nb.size();
    NeighborMat result(n);

    if (lag == 0) {
        for (size_t i = 0; i < n; ++i) {
            result[i] = { i };
        }
    } else if (lag >= n) {
        std::vector<size_t> v(n);
        std::iota(v.begin(), v.end(), size_t{0});
        for (size_t i = 0; i < n; ++i) {
            result[i] = v;
        }
    } else {
        NeighborMat prev = LaggedNeighbors(nb, lag - 1);

        for (size_t i = 0; i < n; ++i) {
            if (prev[i].size() == n) {
                result[i] = prev[i]
            } else {
                std::unordered_set<size_t> merged;
                merged.reserve(prev[i].size() + nb[i].size());
                for (size_t v : prev[i]) {
                    merged.insert(v);
                    for (size_t u : nb[v]) {
                        merged.insert(u);
                    }
                }
                result[i].assign(merged.begin(), merged.end());
                std::sort(result[i].begin(), result[i].end());
            }
        }
    }
    
    return result;
}

/**
 * @brief Extract lagged values from a lattice vector.
 *
 * Each node collects values from its lagged neighbors (no recursively included).
 */
inline Matrix LaggedValues(
    const Vector& vec,
    const NeighborMat& nb,
    size_t lag
) {
    const size_t n = nb.size();
    Matrix out(n);

    if (lag == 0) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = { vec[i] };
        }
        return out;
    }
    
    // Remove duplicates with previous lag (if lag > 1)
    NeighborMat prevNeighbors = LaggedNeighbors(nb, lag-1);
    NeighborMat curNeighbors(n);

    for (size_t i = 0; i < n; ++i) {
        // Convert previous lagged results to a set for fast lookup
        std::unordered_set<size_t> prevSet(prevNeighbors[i].begin(), prevNeighbors[i].end());
        // Remove duplicates from current lagged results
        std::vector<size_t> newIndices;
        for (size_t prev_nb in prevSet){
            for (size_t cur_nb : nb[prev_nb]) {
                if (prevSet.find(cur_nb) == prevSet.end()) {
                    newIndices.push_back(cur_nb);
                }
            }
        }

        // If the new indices are empty, set it to a special value (e.g., std::numeric_limits<int>::min())
        if (newIndices.empty()) {
            newIndices.push_back(std::numeric_limits<int>::min());
        }

        // Update the lagged results
        curNeighbors[i] = newIndices;
    }
    
    for (size_t i = 0; i < n; ++i) {
        out[i].reserve(curNeighbors[i].size());
        for (size_t j :  curNeighbors[i]) {
            out[i].push_back(vec[j]);
        }
    }
    return out;
}

/**
 * @brief Generate lattice embeddings by averaging lagged neighbors.
 *
 * Parameters:
 *   vec    Node values.
 *   nb     Neighbor list.
 *   E      Embedding dimension.
 *   tau    Lag step.
 *   style  0 include current state, otherwise exclude.
 *
 * Columns containing only NaN values are removed automatically.
 */
inline Matrix LatticeEmbedding(
    const Vector& vec,
    const NeighborMat& nb,
    size_t E = 3,
    size_t tau = 1,
    size_t style = 1
) {
    const size_t n = vec.size();
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    Matrix embed(n, Vector(E, NaN));
    std::unordered_map<size_t, NeighborMat> cache;

    auto get_neighbors = [&](size_t lag) -> const NeighborMat& {
        auto it = cache.find(lag);
        if (it != cache.end()) return it->second;
        return cache.emplace(lag, LaggedNeighbors(nb, lag)).first->second;
    };

    size_t start = (style == 0 ? 0 : (tau == 0 ? 0 : tau));
    size_t step  = (tau == 0 ? 1 : tau);
    size_t end   = (tau == 0 ? E - 1 :
                   (style == 0 ? (E - 1) * tau : E * tau));

    for (size_t lag = start; lag <= end; lag += step) {
        const NeighborMat& cur = get_neighbors(lag);
        const size_t col = (lag - start) / step;

        for (size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            size_t cnt = 0;

            for (size_t j : cur[i]) {
                double v = vec[j];
                if (!std::isnan(v)) {
                    sum += v;
                    ++cnt;
                }
            }
            if (cnt > 0) {
                embed[i][col] = sum / cnt;
            }
        }
    }

    /* Remove all-NaN columns */
    std::vector<size_t> validCols;
    for (size_t c = 0; c < embed.front().size(); ++c) {
        bool allNaN = true;
        for (size_t r = 0; r < embed.size(); ++r) {
            if (!std::isnan(embed[r][c])) {
                allNaN = false;
                break;
            }
        }
        if (!allNaN) validCols.push_back(c);
    }

    if (validCols.size() == embed.front().size()) return embed;

    Matrix filtered(n);
    for (size_t r = 0; r < n; ++r) {
        filtered[r].reserve(validCols.size());
        for (size_t c : validCols) {
            filtered[r].push_back(embed[r][c]);
        }
    }
    return filtered;
}

/* =============================================================
 * -------------------- GRID OPERATORS -------------------------
 * ============================================================= */

/**
 * @brief Convert 2D grid coordinates to linear index.
 */
inline size_t GridIndex(size_t row, size_t col, size_t totalCols) {
    return row * totalCols + col;
}

/**
 * @brief Convert linear index to 2D grid coordinates.
 */
inline std::pair<size_t, size_t> GridRowCol(size_t index, size_t totalCols) {
    return { index / totalCols, index % totalCols };
}

/**
 * @brief Flatten a grid matrix into a vector (row major).
 */
inline Vector GridMatToVec(const Matrix& mat) {
    Vector out;
    if (mat.empty()) return out;

    const size_t rows = mat.size();
    const size_t cols = mat.front().size();
    out.reserve(rows * cols);

    for (const auto& row : mat) {
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

/**
 * @brief Reshape a vector into a grid matrix.
 */
inline Matrix GridVecToMat(const Vector& vec, size_t nrow) {
    if (nrow == 0 || vec.size() % nrow != 0) {
        throw std::invalid_argument("GridVecToMat: incompatible dimensions.");
    }

    const size_t ncol = vec.size() / nrow;
    Matrix mat(nrow, Vector(ncol));

    for (size_t i = 0; i < nrow; ++i) {
        for (size_t j = 0; j < ncol; ++j) {
            mat[i][j] = vec[i * ncol + j];
        }
    }
    return mat;
}

/**
 * @brief Compute lagged Moore neighborhood values on a grid.
 */
inline Matrix LaggedGridValues(
    const Matrix& mat,
    size_t lag
) {
    if (mat.empty() || mat.front().empty()) return {};

    const size_t rows = mat.size();
    const size_t cols = mat.front().size();
    const size_t total = rows * cols;
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    if (lag == 0) {
        Matrix out;
        out.reserve(total);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                out.push_back({ mat[i][j] });
            }
        }
        return out;
    }

    std::vector<std::pair<int,int>> offsets;
    for (int dx = -static_cast<int>(lag); dx <= static_cast<int>(lag); ++dx) {
        for (int dy = -static_cast<int>(lag); dy <= static_cast<int>(lag); ++dy) {
            if (std::max(std::abs(dx), std::abs(dy)) == static_cast<int>(lag)) {
                offsets.emplace_back(dx, dy);
            }
        }
    }

    Matrix out(total, Vector(offsets.size(), NaN));

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            const size_t id = GridIndex(r, c, cols);
            for (size_t k = 0; k < offsets.size(); ++k) {
                const int nr = static_cast<int>(r) + offsets[k].first;
                const int nc = static_cast<int>(c) + offsets[k].second;
                if (nr >= 0 && nr < static_cast<int>(rows) &&
                    nc >= 0 && nc < static_cast<int>(cols)) {
                    out[id][k] = mat[nr][nc];
                }
            }
        }
    }
    return out;
}

/**
 * @brief Generate grid spatial embeddings.
 */
inline Matrix GridEmbedding(
    const Matrix& mat,
    size_t E = 3,
    size_t tau = 1,
    Index style = 1
) {
    if (mat.empty() || mat.front().empty() || E == 0) return {};

    const size_t rows = mat.size();
    const size_t cols = mat.front().size();
    const size_t total = rows * cols;
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    Matrix embed(total, Vector(E, NaN));

    auto fill_column = [&](size_t col, size_t lag) {
        Matrix lagged = LaggedGridValues(mat, lag);
        for (size_t i = 0; i < lagged.size(); ++i) {
            double sum = 0.0;
            size_t cnt = 0;
            for (double v : lagged[i]) {
                if (!std::isnan(v)) {
                    sum += v;
                    ++cnt;
                }
            }
            if (cnt > 0) embed[i][col] = sum / cnt;
        }
    };

    if (tau == 0) {
        Vector flat = GridMatToVec(mat);
        for (size_t i = 0; i < total; ++i) embed[i][0] = flat[i];
        for (size_t k = 1; k < E; ++k) fill_column(k, k);
    }
    else if (style == 0) {
        Vector flat = GridMatToVec(mat);
        for (size_t i = 0; i < total; ++i) embed[i][0] = flat[i];
        for (size_t k = 1; k < E; ++k) fill_column(k, k * tau);
    }
    else {
        for (size_t k = 0; k < E; ++k) fill_column(k, (k + 1) * tau);
    }

    /* remove all-NaN columns */
    std::vector<Index> validCols;
    for (size_t c = 0; c < embed.front().size(); ++c) {
        bool allNaN = true;
        for (size_t r = 0; r < embed.size(); ++r) {
            if (!std::isnan(embed[r][c])) {
                allNaN = false;
                break;
            }
        }
        if (!allNaN) validCols.push_back(c);
    }

    if (validCols.size() == embed.front().size()) return embed;

    Matrix filtered(total);
    for (size_t r = 0; r < total; ++r) {
        filtered[r].reserve(validCols.size());
        for (size_t c : validCols) {
            filtered[r].push_back(embed[r][c]);
        }
    }
    return filtered;
}

} // namespace Embed

#endif // EMBED_HPP
