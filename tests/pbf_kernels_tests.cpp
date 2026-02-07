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
    neighbors = testing::get_neighbors_slow<2>(0, positions, h);

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

TEST_F(DensityGridFixture, lambda_computation_different_kernels) {
    using ConstraintKernel = pbf::sph::Poly6<2>;  // Same as Python constraint calculation
    using GradientKernel = pbf::sph::Spikey<2>;   // Same as Python gradient calculation

    float mass = 1.2f;
    float rest_density = 0.95f;
    float epsilon = 1.0e-6f;

    // Get neighbors for particle 0
    std::vector<int> neighbors = testing::get_neighbors_slow<2>(0, positions, h);

    // Compute lambda using different kernels (Poly6 for constraint, Spikey for gradients)
    float lambda = pbf::sph::computeLambda<2, ConstraintKernel, GradientKernel>(
        0, rest_density, mass, h, epsilon, neighbors, positions);

    // Basic validation - lambda should be finite
    EXPECT_TRUE(std::isfinite(lambda));

    // Validate mathematical correctness for the mixed-kernel version
    float constraint = pbf::sph::computeDensityConstraint<2, ConstraintKernel>(
        0, rest_density, mass, h, neighbors, positions);

    std::vector<vec2f> gradients;
    pbf::sph::computeDensityConstraintGradients<2, GradientKernel>(
        0, rest_density, mass, h, neighbors, positions, gradients);

    float sum_grad_sq = 0.0f;
    for (const auto& grad : gradients) {
        sum_grad_sq += grad.dot(grad);
    }
    float expected_lambda = -constraint / (sum_grad_sq + epsilon);

    // Validate that computed lambda matches expected value
    EXPECT_NEAR(lambda, expected_lambda, 1.0e-4f);
}

TEST_F(DensityGridFixture, position_correction_matrix_vector_test) {
    using Kernel = pbf::sph::Spikey<2>;  // Same kernel as Python position correction

    float mass = 1.2f;
    float rest_density = 0.95f;
    float epsilon = 1.0e-6f;

    // Generate random lambda values for testing
    std::vector<float> lambdas(nparticles);
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducible tests
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < nparticles; ++i) {
        lambdas[i] = dis(gen);
    }

    // Construct full gradient matrix: gradc[num_particles * dim][num_particles]
    std::vector<std::vector<float>> gradc(nparticles * 2, std::vector<float>(nparticles, 0.0f));

    // Fill gradient matrix using existing constraint gradient function
    for (int i = 0; i < nparticles; ++i) {
        std::vector<int> neighbors = testing::get_neighbors_slow<2>(i, positions, h);
        std::vector<vec2f> gradients;
        pbf::sph::computeDensityConstraintGradients<2, Kernel>(
            i, rest_density, mass, h, neighbors, positions, gradients);

        // Fill the i-th column of gradc matrix
        // gradients[0] is the self-gradient (grad_i C_i)
        // gradients[k+1] is the gradient w.r.t. neighbor k (grad_j C_i)

        // Add self-gradient (gradient of constraint i w.r.t. particle i)
        gradc[i * 2][i] = gradients[0].x;     // x-component
        gradc[i * 2 + 1][i] = gradients[0].y; // y-component

        // Add neighbor gradients
        for (size_t k = 0; k < neighbors.size(); ++k) {
            int neighbor_index = neighbors[k];
            // Fill the neighbor's position rows in column i
            gradc[neighbor_index * 2][i] = gradients[k + 1].x;     // x-component
            gradc[neighbor_index * 2 + 1][i] = gradients[k + 1].y; // y-component
        }
    }

    // Compute position corrections via matrix-vector product
    std::vector<vec2f> deltap(nparticles);
    for (int i = 0; i < nparticles; ++i) {
        float sum_x = 0.0f, sum_y = 0.0f;
        for (int j = 0; j < nparticles; ++j) {
            sum_x += gradc[i * 2][j] * lambdas[j];
            sum_y += gradc[i * 2 + 1][j] * lambdas[j];
        }
        deltap[i] = vec2f(sum_x, sum_y);
    }

    // Compare with calculatePositionCorrection for each particle
    for (int i = 0; i < nparticles; ++i) {
        std::vector<int> neighbors = testing::get_neighbors_slow<2>(i, positions, h);
        vec2f correction;
        pbf::sph::calculatePositionCorrection<2, Kernel>(
            i, rest_density, mass, h, neighbors, lambdas, positions, correction);

        EXPECT_NEAR(correction.x, deltap[i].x, 1.0e-4f) << "Particle " << i;
        EXPECT_NEAR(correction.y, deltap[i].y, 1.0e-4f) << "Particle " << i;
    }
}
