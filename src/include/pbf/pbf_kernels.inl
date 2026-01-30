namespace pbf::sph {
    template <int Dim, typename Kernel>
    float computeDensityConstraint(int self_index, float rest_density, float mass, float h,
        std::vector<int> const & neighbors,
        std::vector<Vec<Dim>> const & positions) {
            float density = pbf::sph::computeDensity<Dim, Kernel>(self_index, mass, h, neighbors, positions);
            return density / rest_density - 1.0f;
        }

    template <int Dim, typename Kernel>
    void computeDensityConstraintGradients(int self_index,
        float rest_density, float mass, float h,
        std::vector<int> const& neighbors,
        std::vector<Vec<Dim>> const& positions,
        std::vector<Vec<Dim>>& out_gradients) {
            out_gradients.resize(neighbors.size() + 1);

            // Initialize self-gradient to zero
            out_gradients[0] = Vec<Dim>(0.0f, 0.0f);

            // Get self position
            Vec<Dim> const& pi = positions[self_index];

            // Compute gradients for each neighbor
            for (size_t k = 0; k < neighbors.size(); ++k) {
                int neighbor_index = neighbors[k];
                Vec<Dim> const& pj = positions[neighbor_index];

                // Compute displacement vector from self to neighbor
                Vec<Dim> r_vec = pi - pj;

                // Compute kernel derivative using template parameter
                Vec<Dim> gradW = Kernel::deriv(r_vec, h);

                // Neighbor gradient: -(m/ρ₀) * ∇W(rᵢ - rⱼ, h)
                out_gradients[k + 1] = -(mass / rest_density) * gradW;

                // Accumulate self-gradient: (m/ρ₀) * ∇W(rᵢ - rⱼ, h)
                out_gradients[0] += (mass / rest_density) * gradW;
            }
    }

    template <int Dim, typename ConstraintKernel, typename GradientKernel>
    float computeLambda(int self_index,
        float rest_density,
        float mass,
        float h,
        float epsilon,
        std::vector<int> const& neighbors,
        std::vector<Vec<Dim>> const& positions) {
            // Compute density constraint C_i using constraint kernel
            float constraint = computeDensityConstraint<Dim, ConstraintKernel>(
                self_index, rest_density, mass, h, neighbors, positions);

            // Compute constraint gradients using gradient kernel
            std::vector<Vec<Dim>> gradients;
            computeDensityConstraintGradients<Dim, GradientKernel>(
                self_index, rest_density, mass, h, neighbors, positions, gradients);

            // Compute sum of squared gradient magnitudes
            float sum_grad_sq = 0.0f;
            for (const auto& grad : gradients) {
                sum_grad_sq += grad.dot(grad);
            }

            // Compute lambda using the formula: λ = -C / (Σ|∇C|² + ε)
            return -constraint / (sum_grad_sq + epsilon);
    }

    // Backward compatibility overload using same kernel for both
    template <int Dim, typename Kernel>
    float computeLambda(int self_index,
        float rest_density,
        float mass,
        float h,
        float epsilon,
        std::vector<int> const& neighbors,
        std::vector<Vec<Dim>> const& positions) {
            return computeLambda<Dim, Kernel, Kernel>(
                self_index, rest_density, mass, h, epsilon, neighbors, positions);
    }
}
