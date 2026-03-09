#pragma once

#include <vector>
#include "pbf/vec.h"
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

    template <>
    struct Poly6<3> {
        static float eval(Vec<3> const & rVec, const float h);
        static float evalAtZero(const float h);
        static Vec<3> deriv(Vec<3> const & rVec, const float h);
    };

    template <int Dim>
    struct Spikey;

    template <>
    struct Spikey<2> {
        static float eval(Vec<2> const & rVec, const float h);
        static float evalAtZero(const float h);
        static Vec<2> deriv(Vec<2> const & rVec, const float h);
    };

    template <>
    struct Spikey<3> {
        static float eval(Vec<3> const & rVec, const float h);
        static float evalAtZero(const float h);
        static Vec<3> deriv(Vec<3> const & rVec, const float h);
    };

    // Cubic spline kernel with compact support (q <= 1).
    template <int Dim>
    struct CubicSpline;

    template <>
    // 2D cubic spline kernel and gradient.
    struct CubicSpline<2> {
        static float eval(Vec<2> const & rVec, const float h);
        static float evalAtZero(const float h);
        static Vec<2> deriv(Vec<2> const & rVec, const float h);
    };

    template <>
    // 3D cubic spline kernel and gradient.
    struct CubicSpline<3> {
        static float eval(Vec<3> const & rVec, const float h);
        static float evalAtZero(const float h);
        static Vec<3> deriv(Vec<3> const & rVec, const float h);
    };

    template <int Dim, typename Kernel>
    float computeDensity(int self_index, float mass, float h,
        std::vector<int> const & neighbors,
        std::vector<Vec<Dim>> const & positions);

    // Apply XSPH viscosity update to velocities
    template <int Dim, typename Kernel>
    void computeXsphViscosity(float viscosity, float mass, float h,
        std::vector<Vec<Dim>> const& positions,
        std::vector<std::vector<int>> const& neighbors,
        std::vector<Vec<Dim>>& velocities);
}

#include "pbf/sph_kernels.inl"