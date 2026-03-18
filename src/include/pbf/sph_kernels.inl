namespace pbf::sph {
    template <int Dim, typename Kernel>
    float computeDensity(int self_index, float mass, float h,
        std::vector<int> const & neighbors,
        std::vector<Vec<Dim>> const & positions) {
            // Calculate self contribution
            float density = mass * Kernel::evalAtZero(h);
            Vec<Dim> const & pos_i = positions[self_index];
            for (int neighbor : neighbors) {
                // Assumes neighbors contains only fluid particles; boundary density is handled separately.
                Vec<Dim> const & pos_j = positions[neighbor];
                density += mass * Kernel::eval(pos_i - pos_j, h);
            }

            return density;

    }

    template <int Dim, typename Kernel>
    float computeBoundaryPsi(int self_index,
        float h,
        float rest_density,
        float pressure_scale,
        std::vector<int> const& boundary_neighbors,
        std::vector<Vec<Dim>> const& boundary_positions) {
            Vec<Dim> const& pi = boundary_positions[self_index];
            float delta = Kernel::evalAtZero(h);

            for (int neighbor_index : boundary_neighbors) {
                if (neighbor_index == self_index) {
                    continue;
                }
                Vec<Dim> const& pj = boundary_positions[neighbor_index];
                delta += Kernel::eval(pi - pj, h);
            }

            const float volume = 1.0f / delta;
            return pressure_scale * rest_density * volume;
    }

    template <int Dim, typename Kernel>
    void computeXsphViscosity(float viscosity, float mass, float h,
        std::vector<Vec<Dim>> const& positions,
        std::vector<std::vector<int>> const& neighbors,
        std::vector<Vec<Dim>>& velocities) {
        const int num_particles = static_cast<int>(positions.size());
        std::vector<float> densities(num_particles, 0.0f);
        for (int i = 0; i < num_particles; ++i) {
            densities[i] = computeDensity<Dim, Kernel>(i, mass, h, neighbors[i], positions);
        }

        std::vector<Vec<Dim>> deltas(num_particles, Vec<Dim>::zero());
        for (int i = 0; i < num_particles; ++i) {
            const auto& vi = velocities[i];
            const auto& xi = positions[i];
            for (int neighbor : neighbors[i]) {
                const float density_j = densities[neighbor];
                if (density_j <= 0.0f) {
                    continue;
                }
                const auto& vj = velocities[neighbor];
                const auto& xj = positions[neighbor];
                deltas[i] += (vj - vi) * (mass / density_j) * Kernel::eval(xi - xj, h);
            }
        }

        for (int i = 0; i < num_particles; ++i) {
            velocities[i] += viscosity * deltas[i];
        }
    }
}