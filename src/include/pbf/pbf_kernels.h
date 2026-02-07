#pragma once

#include "pbf/vec.h"
#include "pbf/sph_kernels.h"
#include <vector>

namespace pbf::sph {
    template <int Dim, typename Kernel>
    float computeDensityConstraint(int self_index, float restDensity, float mass, float h,
        std::vector<int> const & neighbors,
        std::vector<Vec<Dim>> const & positions);

    template <int Dim, typename Kernel>
    void computeDensityConstraintGradients(
        int self_index,
        float rest_density,
        float mass,
        float h,
        std::vector<int> const& neighbors,
        std::vector<Vec<Dim>> const& positions,
        std::vector<Vec<Dim>>& out_gradients  // output parameter
    );

    template <int Dim, typename ConstraintKernel, typename GradientKernel>
    float computeLambda(int self_index,
        float rest_density,
        float mass,
        float h,
        float epsilon,
        std::vector<int> const& neighbors,
        std::vector<Vec<Dim>> const& positions);

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
}

#include "pbf/pbf_kernels.inl"