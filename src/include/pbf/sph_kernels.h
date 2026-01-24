#pragma once

#include <vector>
#include "pbf/vec2f.h"
// SPH functions used in position based fluids code
namespace pbf::sph {
    template<int Dim>
    struct Poly6;

    template <>
    struct Poly6<2> {
        static float eval(Vec<2> const & rVec, const float h);
        static float evalAtZero(const float h);
        static Vec<2> deriv(Vec<2> const & rVec, const float h);
    };

    template <int Dim>
    struct Spikey;

    template <>
    struct Spikey<2> {
        static float eval(Vec<2> const & rVec, const float h);
        static float evalAtZero(const float h);
        static Vec<2> deriv(Vec<2> const & rVec, const float h);
    };

    template <int Dim, typename Kernel>
    float computeDensity(int self_index, float mass, float h,
        std::vector<int> const & neighbors,
        std::vector<Vec<Dim>> const & positions);
}

#include "pbf/sph_kernels.inl"