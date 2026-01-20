#pragma once

#include "pbf/vec2f.h"
#include "pbf/sph_kernels.h"
#include <vector>

namespace pbf::sph {
    template <int Dim, typename Kernel>
    float computeDensityConstraint(int self_index, float restDensity, float mass, float h,
        std::vector<int> const & neighbors,
        std::vector<Vec<Dim>> const & positions);
}

#include "pbf/pbf_kernels.inl"