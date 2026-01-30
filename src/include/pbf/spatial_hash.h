#pragma once

#include <vector>
#include <unordered_map>
#include <cstdint>
#include "pbf/vec2f.h"

namespace pbf {

/**
 * Spatial hash data structure for efficient neighbor queries in SPH simulations.
 *
 * Divides the simulation domain into a grid of cells with size equal to the kernel radius h.
 * Particles are inserted into appropriate cells, and neighbor queries only need to check
 * the particle's own cell plus the 8 surrounding cells.
 *
 * Performance: O(n) build time, O(1) average query time per particle
 */
class SpatialHash {
public:
    /**
     * Constructor
     * @param xmin Minimum x coordinate of simulation domain
     * @param ymin Minimum y coordinate of simulation domain
     * @param xmax Maximum x coordinate of simulation domain
     * @param ymax Maximum y coordinate of simulation domain
     * @param h Kernel radius (also used as cell size)
     */
    SpatialHash(float xmin, float ymin, float xmax, float ymax, float h);

    /**
     * Update particle positions in the spatial hash
     * @param positions Vector of particle positions
     */
    void update(const std::vector<vec2f>& positions);

    /**
     * Get neighbors for a specific particle
     * @param particle_index Index of the particle to find neighbors for
     * @return Vector of neighbor particle indices
     */
    std::vector<int> getNeighbors(int particle_index) const;

    /**
     * Get neighbors for all particles (batch operation)
     * @return Vector of neighbor lists, one per particle
     */
    std::vector<std::vector<int>> getAllNeighbors() const;

private:
    // Cell coordinate type
    struct CellCoord {
        int x;
        int y;

        bool operator==(const CellCoord& other) const {
            return x == other.x && y == other.y;
        }
    };

    // Hash function for CellCoord
    struct CellCoordHash {
        std::size_t operator()(const CellCoord& coord) const {
            // Simple hash combining x and y coordinates
            return static_cast<std::size_t>(coord.x) * 31 + static_cast<std::size_t>(coord.y);
        }
    };

    // Domain bounds
    float xmin_, ymin_, xmax_, ymax_;
    float h_;

    // Grid parameters
    int nx_cells_;
    int ny_cells_;

    // Cell size (equal to kernel radius)
    float cell_size_;

    // Hash map from cell coordinates to particle indices
    std::unordered_map<CellCoord, std::vector<int>, CellCoordHash> cell_map_;

    // Particle positions and their cell coordinates
    std::vector<vec2f> positions_;
    std::vector<CellCoord> particle_cells_;

    /**
     * Convert world coordinates to cell coordinates
     */
    CellCoord worldToCell(float x, float y) const;

    /**
     * Convert cell coordinates to world coordinates (center of cell)
     */
    vec2f cellToWorld(int cell_x, int cell_y) const;

    /**
     * Get all particles in a 3x3 neighborhood around a cell
     */
    std::vector<int> getNeighborsInNeighborhood(const CellCoord& center_cell) const;

    /**
     * Check if a particle is within distance h of a query point
     */
    bool isWithinRange(const vec2f& query_pos, int particle_index) const;
};

} // namespace pbf