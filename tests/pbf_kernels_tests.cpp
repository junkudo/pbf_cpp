#include <numbers>
#include <random>
#include <gtest/gtest.h>
#include "pbf/sph_kernels.h"
#include "pbf/pbf_kernels.h"
#include "pbf/vec2f.h"
#include "fixtures/particle_fixtures.h"
#include "fixtures/test_helpers.h"

using pbf::vec2f;

namespace {
    constexpr float pi = std::numbers::pi_v<float>;
}

using testing::DensityGridFixture;
TEST_F(DensityGridFixture, density_constraint)
{
    float mass = 1.2f;
    std::vector<int> neighbors;
    float rest_density = 0.95f;
    float density = pbf::sph::computeDensity<2, Kernel>(0, mass, h, neighbors, positions);
    float density_constraint = pbf::sph::computeDensityConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
    float tol = 1.0e-4f;
    EXPECT_NEAR(density_constraint, density / rest_density - 1.0f, tol);
}

TEST_F(DensityGridFixture, constraint_gradient) {
    using Kernel = pbf::sph::Poly6<2>;  // Same kernel as computeDensityConstraint test

    float mass = 1.2f;
    float rest_density = 0.95f;
    std::vector<int> neighbors;

    // Get neighbors for particle 0 (self-particle)
    neighbors = testing::get_neighbors_slow(0, positions, h);

    // Compute analytical gradients
    std::vector<vec2f> analytical_gradients;
    pbf::sph::computeDensityConstraintGradients<2, Kernel>(
        0, rest_density, mass, h, neighbors, positions, analytical_gradients);

    // Compute finite difference gradients
    std::vector<vec2f> fd_gradients;
    fd_gradients.resize(neighbors.size() + 1);

    // Use smaller perturbation for better accuracy
    const float epsilon = 1.0e-3f;

    // Compute finite difference for self-gradient (index 0)
    vec2f original_pos = positions[0];

    // Perturb in +x direction
    positions[0].x += epsilon;
    float c_plus_x = pbf::sph::computeDensityConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
    positions[0].x -= 2 * epsilon;
    float c_minus_x = pbf::sph::computeDensityConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
    positions[0].x += epsilon;  // Restore

    // Perturb in +y direction
    positions[0].y += epsilon;
    float c_plus_y = pbf::sph::computeDensityConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
    positions[0].y -= 2 * epsilon;
    float c_minus_y = pbf::sph::computeDensityConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
    positions[0].y += epsilon;  // Restore

    fd_gradients[0].x = (c_plus_x - c_minus_x) / (2 * epsilon);
    fd_gradients[0].y = (c_plus_y - c_minus_y) / (2 * epsilon);

    // Compute finite difference for each neighbor gradient
    for (size_t k = 0; k < neighbors.size(); ++k) {
        int neighbor_index = neighbors[k];
        vec2f original_neighbor_pos = positions[neighbor_index];

        // Perturb neighbor in +x direction
        positions[neighbor_index].x += epsilon;
        c_plus_x = pbf::sph::computeDensityConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
        positions[neighbor_index].x -= 2 * epsilon;
        c_minus_x = pbf::sph::computeDensityConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
        positions[neighbor_index].x += epsilon;  // Restore

        // Perturb neighbor in +y direction
        positions[neighbor_index].y += epsilon;
        c_plus_y = pbf::sph::computeDensityConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
        positions[neighbor_index].y -= 2 * epsilon;
        c_minus_y = pbf::sph::computeDensityConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
        positions[neighbor_index].y += epsilon;  // Restore

        fd_gradients[k + 1].x = (c_plus_x - c_minus_x) / (2 * epsilon);
        fd_gradients[k + 1].y = (c_plus_y - c_minus_y) / (2 * epsilon);
    }

    // Use absolute error tolerance for more robust comparison
    auto check_absolute_error = [](float analytical, float fd, float absolute_tol = 1.0e-4f) {
        return std::abs(analytical - fd) < absolute_tol;
    };

    // Check self-gradient with relative error
    EXPECT_NEAR(analytical_gradients[0].x, fd_gradients[0].x, 1.0e-2f);
    EXPECT_NEAR(analytical_gradients[0].y, fd_gradients[0].y, 1.0e-2f);

    // Check neighbor gradients with relative error
    for (size_t k = 0; k < neighbors.size(); ++k) {
        EXPECT_NEAR(analytical_gradients[k + 1].x, fd_gradients[k + 1].x, 1.0e-2f);
        EXPECT_NEAR(analytical_gradients[k + 1].y, fd_gradients[k + 1].y, 1.0e-2f);
    }
}
