namespace pbf::sph {
    template <int Dim, typename Kernel>
    float computeDensityConstraint(int self_index, float rest_density, float mass, float h,
        std::vector<int> const & neighbors,
        std::vector<Vec<Dim>> const & positions) {
            float density = pbf::sph::computeDensity<Dim, Kernel>(self_index, mass, h, neighbors, positions);
            return density / rest_density - 1.0f;
        }
}