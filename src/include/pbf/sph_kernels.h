#pragma once

#include <cmath>
#include <numbers>
#include <vector>
#include "pbf/vec.h"
// SPH functions used in position based fluids code
namespace pbf::sph {
    template<int Dim>
    struct Poly6;

    template <>
    struct Poly6<2> {
        explicit Poly6(float h);

        float eval(float r) const;
        float evalAtZero() const;
        float dWdr(float r) const;

    private:
        float h_;
        float h2_;
        float coeff_;
        float eval_at_zero_;
    };

    template <>
    struct Poly6<3> {
        explicit Poly6(float h);

        float eval(float r) const;
        float evalAtZero() const;
        float dWdr(float r) const;

    private:
        float h_;
        float h2_;
        float coeff_;
        float eval_at_zero_;
    };

    template <int Dim>
    struct Spikey;

    template <>
    struct Spikey<2> {
        explicit Spikey(float h);

        float eval(float r) const;
        float evalAtZero() const;
        float dWdr(float r) const;

    private:
        float h_;
        float coeff_;
        float eval_at_zero_;
    };

    template <>
    struct Spikey<3> {
        explicit Spikey(float h);

        float eval(float r) const;
        float evalAtZero() const;
        float dWdr(float r) const;

    private:
        float h_;
        float coeff_;
        float eval_at_zero_;
    };

    // Cubic spline kernel with compact support (q <= 1).
    template <int Dim>
    struct CubicSpline;

    template <>
    // 2D cubic spline kernel and gradient.
    struct CubicSpline<2> {
        explicit CubicSpline(float h);

        float eval(float r) const;
        float evalAtZero() const;
        float dWdr(float r) const;

    private:
        float h_;
        float inv_h_;
        float inv_h2_;
        float k_;
    };

    template <>
    // 3D cubic spline kernel and gradient.
    struct CubicSpline<3> {
        explicit CubicSpline(float h);

        float eval(float r) const;
        float evalAtZero() const;
        float dWdr(float r) const;

    private:
        float h_;
        float inv_h_;
        float k_;
    };

    template <int Dim, typename Kernel>
    float computeDensity(int self_index, float mass, float h,
        std::vector<int> const & neighbors,
        std::vector<Vec<Dim>> const & positions);

    // Compute boundary psi (pressure strength * rest density * boundary volume).
    template <int Dim, typename Kernel>
    float computeBoundaryPsi(int self_index,
        float h,
        float rest_density,
        float pressure_scale,
        std::vector<int> const& boundary_neighbors,
        std::vector<Vec<Dim>> const& boundary_positions);

    // Apply XSPH viscosity update to velocities
    template <int Dim, typename Kernel>
    void computeXsphViscosity(float viscosity, float mass, float h,
        std::vector<Vec<Dim>> const& positions,
        std::vector<std::vector<int>> const& neighbors,
        std::vector<Vec<Dim>>& velocities);
}

#include "pbf/sph_kernels.inl"