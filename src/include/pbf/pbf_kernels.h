#pragma once

#include "pbf/vec.h"
#include "pbf/sph_kernels.h"
#include <vector>

namespace pbf::sph {
    // Fluid-only constraint C_i = rho_i / rho_0 - 1 (no boundary terms).
    template <int Dim, typename Kernel>
    float computeConstraint(int self_index, float restDensity, float mass, float h,
        std::vector<int> const & neighbors,
        std::vector<Vec<Dim>> const & positions);

    // Boundary-only density contribution using precomputed boundary particles.
    template <int Dim, typename Kernel>
    float computeConstraintBoundary(
        int self_index,
        float h,
        std::vector<int> const& boundary_neighbors,
        std::vector<Vec<Dim>> const& positions,
        std::vector<Vec<Dim>> const& boundary_positions,
        std::vector<float> const& boundary_psi);


    // Lambda using both fluid neighbors and boundary particles.
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
        std::vector<float> const& boundary_psi);

    // Fluid-only position correction from lambdas (no boundary contributions).
    template <int Dim, typename GradientKernel>
    void calculatePositionCorrection(
        int self_index,
        float rest_density,
        float mass,
        float h,
        std::vector<int> const& neighbors,
        std::vector<float> const& lambdas,
        std::vector<Vec<Dim>> const& positions,
        Vec<Dim>& out_correction  // output parameter
    );

    // Boundary-only position correction using lambda_i only (FluidDemo style).
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
        Vec<Dim>& out_correction  // output parameter
    );

    namespace detail {
        // Fluid-only constraint gradients dC_i/dx for neighbors (no boundary terms).
        template <int Dim, typename Kernel>
        void computeConstraintGradients(
            int self_index,
            float rest_density,
            float mass,
            float h,
            std::vector<int> const& neighbors,
            std::vector<Vec<Dim>> const& positions,
            std::vector<Vec<Dim>>& out_gradients  // output parameter
        );

        // Boundary-only constraint gradients dC_i/dx for boundary neighbors.
        template <int Dim, typename GradientKernel>
        void computeConstraintGradientsBoundary(
            int self_index,
            float rest_density,
            float h,
            std::vector<int> const& boundary_neighbors,
            std::vector<Vec<Dim>> const& positions,
            std::vector<Vec<Dim>> const& boundary_positions,
            std::vector<float> const& boundary_psi,
            std::vector<Vec<Dim>>& out_gradients  // output parameter
        );
    }
}

#include "pbf/pbf_kernels.inl"