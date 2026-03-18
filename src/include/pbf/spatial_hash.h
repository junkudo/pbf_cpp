#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "pbf/vec.h"

namespace pbf {

/**
 * Spatial hash data structure for efficient neighbor queries in SPH simulations.
 *
 * Divides the simulation domain into a grid of cells with size equal to the kernel radius h.
 * Particles are inserted into appropriate cells, and neighbor queries only need to check
 * a 3^Dim neighborhood around the particle's cell.
 *
 * Performance: O(n) build time, O(1) average query time per particle
 */
template<int Dim>
class SpatialHash {
public:
    using Vec = pbf::Vec<Dim>;
    using CellCoord = std::array<int, Dim>;

    /**
     * Constructor
     * @param min_bounds Minimum coordinates of simulation domain
     * @param max_bounds Maximum coordinates of simulation domain
     * @param h Kernel radius (also used as cell size)
     */
    SpatialHash(const Vec& min_bounds, const Vec& max_bounds, float h);

    /**
     * Update particle positions in the spatial hash
     * @param positions Vector of particle positions
     */
    void update(const std::vector<Vec>& positions);

    /**
     * Get neighbors for a specific particle
     * @param particle_index Index of the particle to find neighbors for
     * @return Vector of neighbor particle indices
     */
    void getNeighbors(int particle_index, std::vector<int>& neighbors) const;

    /**
     * Get neighbors for an arbitrary query position.
     * @param query_pos Position to find neighbors for
     * @return Vector of neighbor particle indices
     */
    void getNeighborsForPosition(const Vec& query_pos, std::vector<int>& neighbors) const;

    /**
     * Get neighbors for all particles (batch operation).
     * @return Vector sized to the number of particles, with one neighbor list per particle
     *         (lists may be empty for isolated particles).
     */
    void getAllNeighbors(std::vector<std::vector<int>>& all_neighbors) const;

private:
    struct CellCoordHash {
        std::size_t operator()(const CellCoord& coord) const {
            std::size_t hash = 0;
            for (int dim = 0; dim < Dim; ++dim) {
                hash = hash * 31u + static_cast<std::size_t>(coord[dim]);
            }
            return hash;
        }
    };

    // Domain bounds
    Vec min_bounds_;
    Vec max_bounds_;
    float h_;

    // Grid parameters
    CellCoord cell_counts_{};

    // Cell size (equal to kernel radius)
    float cell_size_;

    // Hash map from cell coordinates to particle indices
    std::unordered_map<CellCoord, std::vector<int>, CellCoordHash> cell_map_;

    // Particle positions and their cell coordinates
    std::vector<Vec> positions_;
    std::vector<CellCoord> particle_cells_;

    /**
     * Convert world coordinates to cell coordinates
     */
    CellCoord worldToCell(const Vec& position) const;

    /**
     * Convert cell coordinates to world coordinates (center of cell)
     */
    Vec cellToWorld(const CellCoord& cell) const;

    /**
     * Get all particles in a 3^Dim neighborhood around a cell
     */
    void getNeighborsInNeighborhood(const CellCoord& center_cell, std::vector<int>& neighbors) const;

    /**
     * Check if a particle is within distance h of a query point
     */
    bool isWithinRange(const Vec& query_pos, int particle_index) const;
};

} // namespace pbf

#include "pbf/spatial_hash.inl"