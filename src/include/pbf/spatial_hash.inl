namespace pbf {

template<int Dim>
SpatialHash<Dim>::SpatialHash(const Vec& min_bounds, const Vec& max_bounds, float h)
    : min_bounds_(min_bounds), max_bounds_(max_bounds), h_(h), cell_size_(h) {

    for (int dim = 0; dim < Dim; ++dim) {
        float span = max_bounds_[dim] - min_bounds_[dim];
        int cell_count = static_cast<int>(std::ceil(span / cell_size_));
        cell_counts_[dim] = std::max(1, cell_count);
    }
}

template<int Dim>
void SpatialHash<Dim>::update(const std::vector<Vec>& positions) {
    positions_ = positions;
    particle_cells_.resize(positions.size());
    cell_map_.clear();

    for (size_t i = 0; i < positions.size(); ++i) {
        const Vec& pos = positions[i];
        CellCoord cell = worldToCell(pos);
        particle_cells_[i] = cell;
        cell_map_[cell].push_back(static_cast<int>(i));
    }
}

template<int Dim>
std::vector<int> SpatialHash<Dim>::getNeighbors(int particle_index) const {
    if (particle_index < 0 || particle_index >= static_cast<int>(positions_.size())) {
        return {};
    }

    const Vec& query_pos = positions_[particle_index];
    const CellCoord& center_cell = particle_cells_[particle_index];
    std::vector<int> candidates = getNeighborsInNeighborhood(center_cell);

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

template<int Dim>
std::vector<std::vector<int>> SpatialHash<Dim>::getAllNeighbors() const {
    std::vector<std::vector<int>> all_neighbors(positions_.size());

    for (size_t i = 0; i < positions_.size(); ++i) {
        all_neighbors[i] = getNeighbors(static_cast<int>(i));
    }

    return all_neighbors;
}

template<int Dim>
typename SpatialHash<Dim>::CellCoord SpatialHash<Dim>::worldToCell(const Vec& position) const {
    CellCoord cell{};
    for (int dim = 0; dim < Dim; ++dim) {
        float clamped = std::max(min_bounds_[dim], std::min(max_bounds_[dim], position[dim]));
        int cell_index = static_cast<int>((clamped - min_bounds_[dim]) / cell_size_);
        cell_index = std::max(0, std::min(cell_counts_[dim] - 1, cell_index));
        cell[dim] = cell_index;
    }
    return cell;
}

template<int Dim>
typename SpatialHash<Dim>::Vec SpatialHash<Dim>::cellToWorld(const CellCoord& cell) const {
    Vec world_pos = Vec::zero();
    for (int dim = 0; dim < Dim; ++dim) {
        world_pos[dim] = min_bounds_[dim] + (static_cast<float>(cell[dim]) + 0.5f) * cell_size_;
    }
    return world_pos;
}

template<int Dim>
std::vector<int> SpatialHash<Dim>::getNeighborsInNeighborhood(const CellCoord& center_cell) const {
    std::vector<int> neighbors;

    int neighborhood_count = 1;
    for (int dim = 0; dim < Dim; ++dim) {
        neighborhood_count *= 3;
    }

    for (int idx = 0; idx < neighborhood_count; ++idx) {
        int remainder = idx;
        CellCoord neighbor_cell = center_cell;
        bool in_bounds = true;

        for (int dim = 0; dim < Dim; ++dim) {
            int offset = (remainder % 3) - 1;
            remainder /= 3;
            neighbor_cell[dim] += offset;
            if (neighbor_cell[dim] < 0 || neighbor_cell[dim] >= cell_counts_[dim]) {
                in_bounds = false;
                break;
            }
        }

        if (!in_bounds) {
            continue;
        }

        auto it = cell_map_.find(neighbor_cell);
        if (it != cell_map_.end()) {
            const std::vector<int>& cell_particles = it->second;
            neighbors.insert(neighbors.end(), cell_particles.begin(), cell_particles.end());
        }
    }

    return neighbors;
}

template<int Dim>
bool SpatialHash<Dim>::isWithinRange(const Vec& query_pos, int particle_index) const {
    const Vec& particle_pos = positions_[particle_index];
    Vec diff = query_pos - particle_pos;
    float distance_squared = diff.dot(diff);
    return distance_squared <= h_ * h_;
}

} // namespace pbf