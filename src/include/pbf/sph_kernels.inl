namespace pbf::sph {
    template <int Dim, typename Kernel>
    float computeDensity(int self_index, std::vector<int> const & neighbors,
        std::vector<Vec<Dim>> const & positions, float mass, float h) {
            // Calculate self contribution
            float density = mass * Kernel::evalAtZero(h);
            Vec<Dim> const & pos_i = positions[self_index];
            for (int neighbor : neighbors) {
                // TODO - conditional check if boundary particle or not somehow
                // What's the best way to do this?   Is it hacky to use the neighbor ix
                // as a flag?
                Vec<Dim> const & pos_j = positions[neighbor];
                density += mass * Kernel::eval(pos_i - pos_j, h);
            }

            return density;

    }
}