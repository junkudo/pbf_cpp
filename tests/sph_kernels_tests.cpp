#include <numbers>
#include <random>
#include <gtest/gtest.h>
#include "pbf/sph_kernels.h"
#include "pbf/vec2f.h"
#include "pbf/vec3f.h"
#include "fixtures/particle_fixtures.h"
#include "fixtures/test_helpers.h"

namespace {
    constexpr float pi = std::numbers::pi_v<float>;
}

template <typename SPHKernel>
void testSPHKernelNormalization(std::string const & implName) {
SCOPED_TRACE(testing::Message() << "Implementation: " << implName);
    // We expect that the sph kernels integrate to 1 over
    // the circle with radius h
    float h = 3.2f;
    float dr = 0.001f;
    float integral = 0.0f;

    float r = 0.0f;
    pbf::vec2f unitVec = pbf::vec2f(1.0f, 0.0f);
    while (r <= h) {

        float f = SPHKernel::eval(unitVec*r, h);
        integral += 2.0f * pi * r * dr * f;
        r += dr;
    }

    float tolerance = 1.0e-3f;
    EXPECT_NEAR(integral, 1.0f, tolerance);
}

TEST(sph_kernels, normalization)
{
    testSPHKernelNormalization<pbf::sph::Poly6<2>>("Poly6");
    testSPHKernelNormalization<pbf::sph::Spikey<2>>("Spikey");
    // CubicSpline uses support q <= 1 to mirror FluidDemo's neighbor cutoff;
    // the 2D normalization constant is renormalized so the integral is 1.
    testSPHKernelNormalization<pbf::sph::CubicSpline<2>>("CubicSpline");
}

template <typename SPHKernel>
void testSPHKernelNormalization3D(std::string const & implName) {
SCOPED_TRACE(testing::Message() << "Implementation: " << implName);
    // We expect that the sph kernels integrate to 1 over
    // the sphere with radius h
    float h = 3.2f;
    float dr = 0.001f;
    float integral = 0.0f;

    float r = 0.0f;
    pbf::vec3f unitVec = pbf::vec3f(1.0f, 0.0f, 0.0f);
    while (r <= h) {
        float f = SPHKernel::eval(unitVec*r, h);
        integral += 4.0f * pi * r * r * dr * f;
        r += dr;
    }

    float tolerance = 1.0e-3f;
    EXPECT_NEAR(integral, 1.0f, tolerance);
}

TEST(sph_kernels, normalization_3d)
{
    testSPHKernelNormalization3D<pbf::sph::Poly6<3>>("Poly6");
    testSPHKernelNormalization3D<pbf::sph::Spikey<3>>("Spikey");
    testSPHKernelNormalization3D<pbf::sph::CubicSpline<3>>("CubicSpline");
}

template <int Dim, typename SPHKernel>
void testSPHKernelConsistencyAtZero(std::string const & implName, float h) {
SCOPED_TRACE(testing::Message() << "Implementation: " << implName);
    // We expect the sph kernels to evaluate consistently at zero
    float r = 0.0f;
    auto unitVec = pbf::Vec<Dim>::zero();
    unitVec[0] = 1.0f;
    float f = SPHKernel::eval(unitVec*r, h);
    float fAtZero = SPHKernel::evalAtZero(h);
    float tolerance = 1.0e-3f;
    EXPECT_NEAR(f, fAtZero, tolerance);
}


TEST(sph_kernels, poly6_at_zero)
{
    float h = 3.2f;
    testSPHKernelConsistencyAtZero<2, pbf::sph::Poly6<2>>("Poly6", h);
    float fAtZero = pbf::sph::Poly6<2>::evalAtZero(h);
    float tolerance = 1.0e-3f;
    float f =  4.0f / (std::numbers::pi_v<float> * std::pow(h, 8)) * std::pow(h, 6);
    EXPECT_NEAR(f, fAtZero, tolerance);
}

TEST(sph_kernels, spikey_at_zero)
{
    float h = 3.2f;
    testSPHKernelConsistencyAtZero<2, pbf::sph::Spikey<2>>("Spikey", h);
    float fAtZero = pbf::sph::Spikey<2>::evalAtZero(h);
    float tolerance = 1.0e-3f;
    float f =  10.0f / (std::numbers::pi_v<float> * std::pow(h, 5)) * std::pow(h, 3);
    EXPECT_NEAR(f, fAtZero, tolerance);
}

TEST(sph_kernels, cubic_spline_at_zero)
{
    float h = 3.2f;
    testSPHKernelConsistencyAtZero<2, pbf::sph::CubicSpline<2>>("CubicSpline", h);
    float fAtZero = pbf::sph::CubicSpline<2>::evalAtZero(h);
    float tolerance = 1.0e-3f;
    float f = 40.0f / (7.0f * std::numbers::pi_v<float> * std::pow(h, 2));
    EXPECT_NEAR(f, fAtZero, tolerance);
}

TEST(sph_kernels, poly6_at_zero_3d)
{
    float h = 3.2f;
    testSPHKernelConsistencyAtZero<3, pbf::sph::Poly6<3>>("Poly6", h);
    float fAtZero = pbf::sph::Poly6<3>::evalAtZero(h);
    float tolerance = 1.0e-3f;
    float f =  315.0f / (64.0f * std::numbers::pi_v<float> * std::pow(h, 9)) * std::pow(h, 6);
    EXPECT_NEAR(f, fAtZero, tolerance);
}

TEST(sph_kernels, spikey_at_zero_3d)
{
    float h = 3.2f;
    testSPHKernelConsistencyAtZero<3, pbf::sph::Spikey<3>>("Spikey", h);
    float fAtZero = pbf::sph::Spikey<3>::evalAtZero(h);
    float tolerance = 1.0e-3f;
    float f =  15.0f / (std::numbers::pi_v<float> * std::pow(h, 6)) * std::pow(h, 3);
    EXPECT_NEAR(f, fAtZero, tolerance);
}

TEST(sph_kernels, cubic_spline_at_zero_3d)
{
    float h = 3.2f;
    testSPHKernelConsistencyAtZero<3, pbf::sph::CubicSpline<3>>("CubicSpline", h);
    float fAtZero = pbf::sph::CubicSpline<3>::evalAtZero(h);
    float tolerance = 1.0e-3f;
    float f = 8.0f / (std::numbers::pi_v<float> * std::pow(h, 3));
    EXPECT_NEAR(f, fAtZero, tolerance);
}

template <typename SPHKernel>
void testSPHKernelFiniteDifference(std::string const & implName) {
SCOPED_TRACE(testing::Message() << "Implementation: " << implName);
    // We expect the gradients of the kernels to be consistent
    // with the function evaluation
    float h = 3.2f;
    float dr = 0.001f;

    // --------------------------
    // Test a random inner point
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    pbf::vec2f unitVec = pbf::vec2f(dist(testing::rng), dist(testing::rng));
    unitVec /= unitVec.length();

    {
        float r = dist(testing::rng);
        pbf::vec2f rvec = r*unitVec;
        pbf::vec2f dfdr = SPHKernel::deriv(rvec, h);
        float eps = 1.0e-3f;

        for (int dim = 0; dim < 2; ++dim) {
            pbf::vec2f rvec_plus = rvec;
            rvec_plus[dim] += eps;
            float fplus = SPHKernel::eval(rvec_plus, h);

            pbf::vec2f rvec_minus = rvec;
            rvec_minus[dim] -= eps;
            float fminus = SPHKernel::eval(rvec_minus, h);

            float dfdx = (fplus - fminus) / (2.0f * eps);
            float tolerance = 1.0e-4f;
            EXPECT_NEAR(dfdx, dfdr[dim], tolerance);
        }
    }

    // --------------------------
    // Test at boundaries
    { // Testing at r = 0.0
        float r = 0.0f;
        pbf::vec2f rvec = r*unitVec;
        pbf::vec2f dfdr = SPHKernel::deriv(rvec, h);
        float tolerance = 1.0e-4f;
        for (int dim = 0; dim < 2; ++dim) {
            EXPECT_NEAR(0.0f, dfdr[dim], tolerance);
        }
    }

    { // Testing at r = h
        float r = h;
        pbf::vec2f rvec = r*unitVec;
        pbf::vec2f dfdr = SPHKernel::deriv(rvec, h);
        float tolerance = 1.0e-4f;
        for (int dim = 0; dim < 2; ++dim) {
            EXPECT_NEAR(0.0f, dfdr[dim], tolerance);
        }
    }
}

TEST(sph_kernels, derivative_consistency) {
    testSPHKernelFiniteDifference<pbf::sph::Poly6<2>>("Poly6");
    testSPHKernelFiniteDifference<pbf::sph::Spikey<2>>("Spikey");
    testSPHKernelFiniteDifference<pbf::sph::CubicSpline<2>>("CubicSpline");
}

template <typename SPHKernel>
void testSPHKernelFiniteDifference3D(std::string const & implName) {
SCOPED_TRACE(testing::Message() << "Implementation: " << implName);
    // We expect the gradients of the kernels to be consistent
    // with the function evaluation
    float h = 3.2f;

    // --------------------------
    // Test a random inner point
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    pbf::vec3f unitVec = pbf::vec3f(dist(testing::rng), dist(testing::rng), dist(testing::rng));
    unitVec /= unitVec.length();

    {
        float r = dist(testing::rng);
        pbf::vec3f rvec = r*unitVec;
        pbf::vec3f dfdr = SPHKernel::deriv(rvec, h);
        float eps = 1.0e-3f;

        for (int dim = 0; dim < 3; ++dim) {
            pbf::vec3f rvec_plus = rvec;
            rvec_plus[dim] += eps;
            float fplus = SPHKernel::eval(rvec_plus, h);

            pbf::vec3f rvec_minus = rvec;
            rvec_minus[dim] -= eps;
            float fminus = SPHKernel::eval(rvec_minus, h);

            float dfdx = (fplus - fminus) / (2.0f * eps);
            float tolerance = 1.0e-4f;
            EXPECT_NEAR(dfdx, dfdr[dim], tolerance);
        }
    }

    // --------------------------
    // Test at boundaries
    { // Testing at r = 0.0
        float r = 0.0f;
        pbf::vec3f rvec = r*unitVec;
        pbf::vec3f dfdr = SPHKernel::deriv(rvec, h);
        float tolerance = 1.0e-4f;
        for (int dim = 0; dim < 3; ++dim) {
            EXPECT_NEAR(0.0f, dfdr[dim], tolerance);
        }
    }

    { // Testing at r = h
        float r = h;
        pbf::vec3f rvec = r*unitVec;
        pbf::vec3f dfdr = SPHKernel::deriv(rvec, h);
        float tolerance = 1.0e-4f;
        for (int dim = 0; dim < 3; ++dim) {
            EXPECT_NEAR(0.0f, dfdr[dim], tolerance);
        }
    }
}

TEST(sph_kernels, derivative_consistency_3d) {
    testSPHKernelFiniteDifference3D<pbf::sph::Poly6<3>>("Poly6");
    testSPHKernelFiniteDifference3D<pbf::sph::Spikey<3>>("Spikey");
    testSPHKernelFiniteDifference3D<pbf::sph::CubicSpline<3>>("CubicSpline");
}

using testing::DensityGridFixture;
TEST_F(DensityGridFixture, density_one_particle) {
    // This tests the single particle case which should just be
    // mass * Kernel evaluated at zero
    float mass = 2.3f;
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<pbf::vec2f> positions;
    positions.push_back({uni(testing::rng), uni(testing::rng)});
    std::vector<int> neighbors; // empty neighbors
    float tol = 1.0e-4f;
    float density0 = pbf::sph::computeDensity<2, Kernel>(0, mass, h, neighbors, positions);
    EXPECT_NEAR(density0, mass * Kernel::evalAtZero(h), tol);

    float density1 = pbf::sph::computeDensity<2, Kernel>(0, mass * 2.0f, h, neighbors, positions);
    EXPECT_NEAR(density1, 2.0f * mass * Kernel::evalAtZero(h), tol);
}

TEST_F(DensityGridFixture, density_two_colocated_particles) {
    // This tests two particles that are colocated (which should just be double the calculation)
    float mass = 2.3f;
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<pbf::vec2f> positions;
    positions.push_back({uni(testing::rng), uni(testing::rng)});
    positions.push_back(positions[0]);

    std::vector<int> neighbors {1};

    float density = pbf::sph::computeDensity<2, Kernel>(0, mass, h, neighbors, positions);
    float tol = 1.0e-4f;
    EXPECT_NEAR(density, 2.0f * mass * Kernel::evalAtZero(h), tol);
}

TEST_F(DensityGridFixture, density_three_particles) {
    // This tests density with 2 neighbors at known distances
    float mass = 2.3f;
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<pbf::vec2f> positions;
    positions.push_back({uni(testing::rng), uni(testing::rng)});

    pbf::vec2f direction(uni(testing::rng), uni(testing::rng));
    direction /= direction.length();

    float r0 = 0.50f * h;
    float r1 = 0.75f * h;
    positions.push_back(positions[0] + direction * r0);
    positions.push_back(positions[0] + direction * r1);
    std::vector<int> neighbors{1,2};

    float density = pbf::sph::computeDensity<2, Kernel>(0, mass, h, neighbors, positions);
    float expectedDensity = mass * (Kernel::evalAtZero(h) + Kernel::eval(direction * r0, h) + Kernel::eval(direction * r1, h));
    float tol = 1.0e-4f;
    EXPECT_NEAR(density, expectedDensity, tol);
}

TEST_F(DensityGridFixture, density_geometric_invariants) {
    std::vector<float> densities(nparticles);
    float mass = 1.2f;
    for (int i = 0; i < nparticles; ++i) {
        std::vector<int> neighbors = testing::get_neighbors_slow<2>(i, positions, h);
        float density = pbf::sph::computeDensity<2, Kernel>(i,  mass, h, neighbors, positions);
        densities.at(i) = density;
    }

    // Test translational invariance
    auto new_positions = positions;
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    pbf::vec2f translation(uni(testing::rng), uni(testing::rng));

    for (auto & position : new_positions) {
        position += translation;
    }

    for (int i = 0; i < nparticles; ++i) {
        std::vector<int> neighbors = testing::get_neighbors_slow<2>(i, new_positions, h);
        float density = pbf::sph::computeDensity<2, Kernel>(i, mass, h, neighbors, new_positions);
        float tol = 1.0e-4f;
        EXPECT_NEAR(density, densities.at(i), tol);
    }

    // Test rotational invariance
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * static_cast<float>(std::numbers::pi));
    float angle = angle_dist(testing::rng);

    float c = std::cos(angle);
    float s = std::sin(angle);

    auto rotated_positions = positions;

    for (auto& p : rotated_positions) {
        float x_new = c * p.x - s * p.y;
        float y_new = s * p.x + c * p.y;
        p.x = x_new;
        p.y = y_new;
    }
    for (int i = 0; i < nparticles; ++i) {
        std::vector<int> neighbors = testing::get_neighbors_slow<2>(i, rotated_positions, h);
        float density = pbf::sph::computeDensity<2, Kernel>(i, mass, h, neighbors, rotated_positions);
        float tol = 1.0e-4f;
        EXPECT_NEAR(density, densities.at(i), tol);
    }
}

using testing::DensityGridFixture3D;
TEST_F(DensityGridFixture3D, density_one_particle_3d) {
    // This tests the single particle case which should just be
    // mass * Kernel evaluated at zero
    float mass = 2.3f;
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<pbf::vec3f> positions;
    positions.push_back({uni(testing::rng), uni(testing::rng), uni(testing::rng)});
    std::vector<int> neighbors; // empty neighbors
    float tol = 1.0e-4f;
    float density0 = pbf::sph::computeDensity<3, Kernel>(0, mass, h, neighbors, positions);
    EXPECT_NEAR(density0, mass * Kernel::evalAtZero(h), tol);

    float density1 = pbf::sph::computeDensity<3, Kernel>(0, mass * 2.0f, h, neighbors, positions);
    EXPECT_NEAR(density1, 2.0f * mass * Kernel::evalAtZero(h), tol);
}

TEST_F(DensityGridFixture3D, density_two_colocated_particles_3d) {
    // This tests two particles that are colocated (which should just be double the calculation)
    float mass = 2.3f;
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<pbf::vec3f> positions;
    positions.push_back({uni(testing::rng), uni(testing::rng), uni(testing::rng)});
    positions.push_back(positions[0]);

    std::vector<int> neighbors {1};

    float density = pbf::sph::computeDensity<3, Kernel>(0, mass, h, neighbors, positions);
    float tol = 1.0e-4f;
    EXPECT_NEAR(density, 2.0f * mass * Kernel::evalAtZero(h), tol);
}

TEST_F(DensityGridFixture3D, density_three_particles_3d) {
    // This tests density with 2 neighbors at known distances
    float mass = 2.3f;
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<pbf::vec3f> positions;
    positions.push_back({uni(testing::rng), uni(testing::rng), uni(testing::rng)});

    pbf::vec3f direction(uni(testing::rng), uni(testing::rng), uni(testing::rng));
    direction /= direction.length();

    float r0 = 0.50f * h;
    float r1 = 0.75f * h;
    positions.push_back(positions[0] + direction * r0);
    positions.push_back(positions[0] + direction * r1);
    std::vector<int> neighbors{1,2};

    float density = pbf::sph::computeDensity<3, Kernel>(0, mass, h, neighbors, positions);
    float expectedDensity = mass * (Kernel::evalAtZero(h) + Kernel::eval(direction * r0, h) + Kernel::eval(direction * r1, h));
    float tol = 1.0e-4f;
    EXPECT_NEAR(density, expectedDensity, tol);
}

TEST_F(DensityGridFixture3D, density_geometric_invariants_3d) {
    std::vector<float> densities(nparticles);
    float mass = 1.2f;
    for (int i = 0; i < nparticles; ++i) {
        std::vector<int> neighbors = testing::get_neighbors_slow<3>(i, positions, h);
        float density = pbf::sph::computeDensity<3, Kernel>(i,  mass, h, neighbors, positions);
        densities.at(i) = density;
    }

    // Test translational invariance
    auto new_positions = positions;
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    pbf::vec3f translation(uni(testing::rng), uni(testing::rng), uni(testing::rng));

    for (auto & position : new_positions) {
        position += translation;
    }

    for (int i = 0; i < nparticles; ++i) {
        std::vector<int> neighbors = testing::get_neighbors_slow<3>(i, new_positions, h);
        float density = pbf::sph::computeDensity<3, Kernel>(i, mass, h, neighbors, new_positions);
        float tol = 1.0e-4f;
        EXPECT_NEAR(density, densities.at(i), tol);
    }

    // Test rotational invariance (rotate about z axis)
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * static_cast<float>(std::numbers::pi));
    float angle = angle_dist(testing::rng);

    float c = std::cos(angle);
    float s = std::sin(angle);

    auto rotated_positions = positions;

    for (auto& p : rotated_positions) {
        float x_new = c * p.x - s * p.y;
        float y_new = s * p.x + c * p.y;
        p.x = x_new;
        p.y = y_new;
    }
    for (int i = 0; i < nparticles; ++i) {
        std::vector<int> neighbors = testing::get_neighbors_slow<3>(i, rotated_positions, h);
        float density = pbf::sph::computeDensity<3, Kernel>(i, mass, h, neighbors, rotated_positions);
        float tol = 1.0e-4f;
        EXPECT_NEAR(density, densities.at(i), tol);
    }
}

TEST(sph_kernels, xsph_viscosity_smooths_velocity) {
    using Kernel = pbf::sph::CubicSpline<2>;
    const float h = 1.0f;
    const float mass = 2.0f;
    const float viscosity = 0.02f;

    std::vector<pbf::vec2f> positions{
        pbf::vec2f(0.0f, 0.0f),
        pbf::vec2f(0.5f, 0.0f)
    };
    std::vector<pbf::vec2f> velocities{
        pbf::vec2f(1.0f, 0.0f),
        pbf::vec2f(-1.0f, 0.0f)
    };
    std::vector<std::vector<int>> neighbors{{1}, {0}};

    const float density0 = pbf::sph::computeDensity<2, Kernel>(0, mass, h, neighbors[0], positions);
    const float density1 = pbf::sph::computeDensity<2, Kernel>(1, mass, h, neighbors[1], positions);
    const float weight = Kernel::eval(positions[0] - positions[1], h);
    const pbf::vec2f delta0 = -(velocities[0] - velocities[1]) * (mass / density1) * weight;
    const pbf::vec2f delta1 = -(velocities[1] - velocities[0]) * (mass / density0) * weight;
    const pbf::vec2f expected0 = velocities[0] + viscosity * delta0;
    const pbf::vec2f expected1 = velocities[1] + viscosity * delta1;

    pbf::sph::computeXsphViscosity<2, Kernel>(viscosity, mass, h, positions, neighbors, velocities);

    const float tol = 1.0e-4f;
    EXPECT_NEAR(velocities[0].x, expected0.x, tol);
    EXPECT_NEAR(velocities[0].y, expected0.y, tol);
    EXPECT_NEAR(velocities[1].x, expected1.x, tol);
    EXPECT_NEAR(velocities[1].y, expected1.y, tol);
    EXPECT_LT(std::abs(velocities[0].x), 1.0f);
    EXPECT_LT(std::abs(velocities[1].x), 1.0f);
}

TEST(sph_kernels, xsph_viscosity_zero_no_change) {
    using Kernel = pbf::sph::CubicSpline<2>;
    const float h = 1.0f;
    const float mass = 2.0f;
    const float viscosity = 0.0f;

    std::vector<pbf::vec2f> positions{
        pbf::vec2f(0.0f, 0.0f),
        pbf::vec2f(0.5f, 0.0f)
    };
    std::vector<pbf::vec2f> velocities{
        pbf::vec2f(1.0f, 0.0f),
        pbf::vec2f(-1.0f, 0.0f)
    };
    std::vector<std::vector<int>> neighbors{{1}, {0}};

    const auto expected0 = velocities[0];
    const auto expected1 = velocities[1];

    pbf::sph::computeXsphViscosity<2, Kernel>(viscosity, mass, h, positions, neighbors, velocities);

    const float tol = 1.0e-6f;
    EXPECT_NEAR(velocities[0].x, expected0.x, tol);
    EXPECT_NEAR(velocities[0].y, expected0.y, tol);
    EXPECT_NEAR(velocities[1].x, expected1.x, tol);
    EXPECT_NEAR(velocities[1].y, expected1.y, tol);
}
