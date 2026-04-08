namespace pbf::sph {
    template <int Dim>
    void logVecComponents(const Vec<Dim>& value) {
        if constexpr (Dim == 2) {
            std::cout << value.x << ", " << value.y;
        } else {
            std::cout << value.x << ", " << value.y << ", " << value.z;
        }
    }
    template <int Dim, typename Kernel>
    float computeConstraint(int self_index, float rest_density, float mass, float h,
        std::vector<int> const & neighbors,
        std::vector<Vec<Dim>> const & positions) {
            float density = pbf::sph::computeDensity<Dim, Kernel>(self_index, mass, h, neighbors, positions);
            return density / rest_density - 1.0f;
        }

    namespace detail {
        template <int Dim, typename Kernel>
        Vec<Dim> computeKernelGradient(Kernel const& kernel, Vec<Dim> const& r_vec, float h2) {
            const float r2 = r_vec.dot(r_vec);
            if (r2 > h2) {
                return Vec<Dim>::zero();
            }
            if (r2 <= 1.0e-12f) {
                return Vec<Dim>::zero();
            }
            const float r = std::sqrt(r2);
            const float dWdr = kernel.dWdr(r);
            return r_vec * (dWdr / r);
        }

        template <int Dim, typename Kernel>
        void computeConstraintGradients(int self_index,
            float rest_density, float mass, float h,
            std::vector<int> const& neighbors,
            std::vector<Vec<Dim>> const& positions,
            std::vector<Vec<Dim>>& out_gradients) {
                const Kernel kernel(h);
                const float h2 = h * h;
                out_gradients.resize(neighbors.size() + 1);

                // Initialize self-gradient to zero
                out_gradients[0] = Vec<Dim>::zero();

                // Get self position
                Vec<Dim> const& pi = positions[self_index];

                // Compute gradients for each neighbor
                for (size_t k = 0; k < neighbors.size(); ++k) {
                    int neighbor_index = neighbors[k];
                    Vec<Dim> const& pj = positions[neighbor_index];

                    // Compute displacement vector from self to neighbor
                    Vec<Dim> r_vec = pi - pj;

                    // Compute kernel derivative using template parameter
                    Vec<Dim> gradW = computeKernelGradient<Dim>(kernel, r_vec, h2);

                    // Neighbor gradient: -(m/ρ₀) * ∇W(rᵢ - rⱼ, h)
                    out_gradients[k + 1] = -(mass / rest_density) * gradW;

                    // Accumulate self-gradient: (m/ρ₀) * ∇W(rᵢ - rⱼ, h)
                    out_gradients[0] += (mass / rest_density) * gradW;
                }
        }

        template <int Dim, typename GradientKernel>
        void computeConstraintGradientsBoundary(
            int self_index,
            float rest_density,
            float h,
            std::vector<int> const& boundary_neighbors,
            std::vector<Vec<Dim>> const& positions,
            std::vector<Vec<Dim>> const& boundary_positions,
            std::vector<float> const& boundary_psi,
            std::vector<Vec<Dim>>& out_gradients) {
                const GradientKernel kernel(h);
                const float h2 = h * h;
                // Boundary analog of computeConstraintGradients.
                out_gradients.resize(boundary_neighbors.size());
                Vec<Dim> const& pi = positions[self_index];

                for (size_t k = 0; k < boundary_neighbors.size(); ++k) {
                    int boundary_index = boundary_neighbors[k];
                    Vec<Dim> const& pb = boundary_positions[boundary_index];
                    Vec<Dim> gradW = computeKernelGradient<Dim>(kernel, pi - pb, h2);
                    Vec<Dim> gradC_j = -(boundary_psi[boundary_index] / rest_density) * gradW;
                    out_gradients[k] = gradC_j;
                }
        }
    }

    template <int Dim, typename ConstraintKernel, typename GradientKernel>
    float computeLambdaWithBoundary(
        int self_index,
        float rest_density,
        float mass,
        float h,
        float epsilon,
        std::vector<int> const& neighbors,
        std::vector<int> const& boundary_neighbors,
        std::vector<Vec<Dim>> const& positions,
        std::vector<Vec<Dim>> const& boundary_positions,
        std::vector<float> const& boundary_psi) {
            const GradientKernel gradient_kernel(h);
            const float h2 = h * h;
            // Combine fluid density and boundary density contributions.
            float density = computeDensity<Dim, ConstraintKernel>(
                self_index, mass, h, neighbors, positions);
            density += computeConstraintBoundary<Dim, ConstraintKernel>(
                self_index, h, boundary_neighbors, positions, boundary_positions, boundary_psi);
            float constraint = density / rest_density - 1.0f;

            // Clamp negative density constraints to zero to match FluidDemo and avoid surface clumping.
            constraint = std::max(constraint, 0.0f);

            const float mass_over_rest = mass / rest_density;
            const Vec<Dim>& pi = positions[self_index];

            Vec<Dim> grad_i_fluid = Vec<Dim>::zero();
            float fluid_sum_grad_sq = 0.0f;
            for (int neighbor_index : neighbors) {
                const Vec<Dim>& pj = positions[neighbor_index];
                const Vec<Dim> r_vec = pi - pj;
                const Vec<Dim> gradW = detail::computeKernelGradient<Dim>(gradient_kernel, r_vec, h2);
                const Vec<Dim> grad_j = -(mass_over_rest) * gradW;
                grad_i_fluid -= grad_j;
                fluid_sum_grad_sq += grad_j.dot(grad_j);
            }
            fluid_sum_grad_sq += grad_i_fluid.dot(grad_i_fluid);

            Vec<Dim> grad_i_boundary = Vec<Dim>::zero();
            float boundary_sum_grad_sq = 0.0f;
            for (int boundary_index : boundary_neighbors) {
                const Vec<Dim>& pb = boundary_positions[boundary_index];
                const Vec<Dim> gradW = detail::computeKernelGradient<Dim>(gradient_kernel, pi - pb, h2);
                const Vec<Dim> gradC_j = -(boundary_psi[boundary_index] / rest_density) * gradW;
                grad_i_boundary -= gradC_j;
                boundary_sum_grad_sq += gradC_j.dot(gradC_j);
            }
            boundary_sum_grad_sq += grad_i_boundary.dot(grad_i_boundary);

            const float grad_i_fluid_sq = grad_i_fluid.dot(grad_i_fluid);
            const float boundary_grad_i_sq = grad_i_boundary.dot(grad_i_boundary);
            const Vec<Dim> combined_grad_i = grad_i_fluid + grad_i_boundary;
            const float total_grad_sq = fluid_sum_grad_sq - grad_i_fluid_sq
                                        + (boundary_sum_grad_sq - boundary_grad_i_sq)
                                        + combined_grad_i.dot(combined_grad_i);

            return (constraint > 0.0f)
                       ? -constraint / (total_grad_sq + epsilon)
                       : 0.0f;
    }

    template <int Dim, typename GradientKernel>
    void calculatePositionCorrection(
        int self_index,
        float rest_density,
        float mass,
        float h,
        std::vector<int> const& neighbors,
        std::vector<float> const& lambdas,
        std::vector<Vec<Dim>> const& positions,
        Vec<Dim>& out_correction) {
            const GradientKernel kernel(h);
            const float h2 = h * h;
            // Initialize correction to zero
            out_correction = Vec<Dim>::zero();

            // Get self position
            Vec<Dim> const& pi = positions[self_index];
            float lambda_i = lambdas[self_index];

            // Calculate position correction for each neighbor
            for (size_t k = 0; k < neighbors.size(); ++k) {
                int neighbor_index = neighbors[k];

                Vec<Dim> const& pj = positions[neighbor_index];
                float lambda_j = lambdas[neighbor_index];

                // Compute displacement vector from self to neighbor
                Vec<Dim> r_vec = pi - pj;

                // Compute kernel gradient using template parameter
                Vec<Dim> gradW = detail::computeKernelGradient<Dim>(kernel, r_vec, h2);

                // Calculate position correction contribution:
                // (m/ρ₀) * (λ_i + λ_j) * ∇W(r_i - r_j, h)
                float correction_factor = (mass / rest_density) * (lambda_i + lambda_j);
                out_correction += correction_factor * gradW;
            }
    }

    template <int Dim, typename Kernel>
    float computeConstraintBoundary(
        int self_index,
        float h,
        std::vector<int> const& boundary_neighbors,
        std::vector<Vec<Dim>> const& positions,
        std::vector<Vec<Dim>> const& boundary_positions,
        std::vector<float> const& boundary_psi) {
            const Kernel kernel(h);
            const float h2 = h * h;
            // Sum psi * W for all boundary neighbors.
            float density = 0.0f;
            Vec<Dim> const& pi = positions[self_index];

            for (int boundary_index : boundary_neighbors) {
                Vec<Dim> const& pb = boundary_positions[boundary_index];
                const Vec<Dim> r_vec = pi - pb;
                const float r2 = r_vec.dot(r_vec);
                if (r2 > h2) {
                    continue;
                }
                const float w = kernel.eval(std::sqrt(r2));
                const float psi = boundary_psi[boundary_index];
                density += psi * w;
            }

            return density;
    }

    template <int Dim, typename GradientKernel>
    void computePositionCorrectionBoundary(
        int self_index,
        float rest_density,
        float h,
        std::vector<int> const& boundary_neighbors,
        std::vector<Vec<Dim>> const& positions,
        std::vector<Vec<Dim>> const& boundary_positions,
        std::vector<float> const& boundary_psi,
        float lambda_i,
        Vec<Dim>& out_correction) {
            const GradientKernel kernel(h);
            const float h2 = h * h;
            // FluidDemo-style boundary correction using only lambda_i.
            Vec<Dim> const& pi = positions[self_index];

            for (int boundary_index : boundary_neighbors) {
                Vec<Dim> const& pb = boundary_positions[boundary_index];
                Vec<Dim> gradW = detail::computeKernelGradient<Dim>(kernel, pi - pb, h2);
                Vec<Dim> gradC_j = -(boundary_psi[boundary_index] / rest_density) * gradW;
                out_correction -= lambda_i * gradC_j;
            }
    }
}
