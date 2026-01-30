#include "pbf/spatial_hash.h"
#include <algorithm>
#include <cmath>

namespace pbf {

SpatialHash::SpatialHash(float xmin, float ymin, float xmax, float ymax, float h)
    : xmin_(xmin), ymin_(ymin), xmax_(xmax), ymax_(ymax), h_(h) {

    cell_size_ = h_;

    // Calculate number of cells in each dimension
    nx_cells_ = static_cast<int>(std::ceil((xmax_ - xmin_) / cell_size_));
    ny_cells_ = static_cast<int>(std::ceil((ymax_ - ymin_) / cell_size_));

    // Ensure at least 1 cell in each dimension
    nx_cells_ = std::max(1, nx_cells_);
    ny_cells_ = std::max(1, ny_cells_);
}

void SpatialHash::update(const std::vector<vec2f>& positions) {
    positions_ = positions;
    particle_cells_.resize(positions.size());
    cell_map_.clear();

    // Insert particles into cells
    for (size_t i = 0; i < positions.size(); ++i) {
        const vec2f& pos = positions[i];
        CellCoord cell = worldToCell(pos.x, pos.y);
        particle_cells_[i] = cell;

        // Insert particle index into the appropriate cell
        cell_map_[cell].push_back(static_cast<int>(i));
    }
}

std::vector<int> SpatialHash::getNeighbors(int particle_index) const {
    if (particle_index < 0 || particle_index >= static_cast<int>(positions_.size())) {
        return {};
    }

    const vec2f& query_pos = positions_[particle_index];
    const CellCoord& center_cell = particle_cells_[particle_index];

    // Get all particles in the 3x3 neighborhood
    std::vector<int> candidates = getNeighborsInNeighborhood(center_cell);

    // Filter candidates to only include those within distance h
    std::vector<int> neighbors;
    neighbors.reserve(candidates.size());

    for (int candidate_index : candidates) {
        if (candidate_index != particle_index &&
            isWithinRange(query_pos, candidate_index)) {
            neighbors.push_back(candidate_index);
        }
    }

    return neighbors;
}

std::vector<std::vector<int>> SpatialHash::getAllNeighbors() const {
    std::vector<std::vector<int>> all_neighbors(positions_.size());

    for (size_t i = 0; i < positions_.size(); ++i) {
        all_neighbors[i] = getNeighbors(static_cast<int>(i));
    }

    return all_neighbors;
}

SpatialHash::CellCoord SpatialHash::worldToCell(float x, float y) const {
    // Clamp coordinates to domain bounds
    x = std::max(xmin_, std::min(xmax_, x));
    y = std::max(ymin_, std::min(ymax_, y));

    // Convert to cell coordinates
    int cell_x = static_cast<int>((x - xmin_) / cell_size_);
    int cell_y = static_cast<int>((y - ymin_) / cell_size_);

    // Ensure cell coordinates are within bounds
    cell_x = std::max(0, std::min(nx_cells_ - 1, cell_x));
    cell_y = std::max(0, std::min(ny_cells_ - 1, cell_y));

    return {cell_x, cell_y};
}

vec2f SpatialHash::cellToWorld(int cell_x, int cell_y) const {
    float x = xmin_ + (cell_x + 0.5f) * cell_size_;
    float y = ymin_ + (cell_y + 0.5f) * cell_size_;
    return {x, y};
}

std::vector<int> SpatialHash::getNeighborsInNeighborhood(const CellCoord& center_cell) const {
    std::vector<int> neighbors;

    // Check 3x3 neighborhood around the center cell
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            CellCoord neighbor_cell = {center_cell.x + dx, center_cell.y + dy};

            // Check if neighbor cell is within bounds
            if (neighbor_cell.x >= 0 && neighbor_cell.x < nx_cells_ &&
                neighbor_cell.y >= 0 && neighbor_cell.y < ny_cells_) {

                // Add all particles in this cell
                auto it = cell_map_.find(neighbor_cell);
                if (it != cell_map_.end()) {
                    const std::vector<int>& cell_particles = it->second;
                    neighbors.insert(neighbors.end(), cell_particles.begin(), cell_particles.end());
                }
            }
        }
    }

    return neighbors;
}

bool SpatialHash::isWithinRange(const vec2f& query_pos, int particle_index) const {
    const vec2f& particle_pos = positions_[particle_index];
    vec2f diff = query_pos - particle_pos;
    float distance_squared = diff.dot(diff);
    return distance_squared <= h_ * h_;
}

} // namespace pbf