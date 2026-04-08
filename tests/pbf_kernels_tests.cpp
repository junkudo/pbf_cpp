#include <numbers>
#include <random>
#include <gtest/gtest.h>
#include "pbf/sph_kernels.h"
#include "pbf/pbf_kernels.h"
#include "pbf/vec2f.h"
#include "pbf/vec3f.h"
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
    float density_constraint = pbf::sph::computeConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
    float tol = 1.0e-4f;
    EXPECT_NEAR(density_constraint, density / rest_density - 1.0f, tol);
}

TEST_F(DensityGridFixture, boundary_density_contribution) {
    using Kernel = pbf::sph::Poly6<2>;
    Kernel kernel(h);

    std::vector<vec2f> boundary_positions;
    boundary_positions.push_back(positions[0] + vec2f(0.5f, 0.0f));
    std::vector<float> boundary_psi{1.5f};
    std::vector<int> boundary_neighbors{0};

    float density = pbf::sph::computeConstraintBoundary<2, Kernel>(
        0, h, boundary_neighbors, positions, boundary_positions, boundary_psi);
    float expected_density = boundary_psi[0] * kernel.eval(positions[0] - boundary_positions[0]);

    EXPECT_NEAR(density, expected_density, 1.0e-6f);
}

TEST_F(DensityGridFixture, boundary_gradient_sum_sq) {
    using GradientKernel = pbf::sph::Spikey<2>;
    GradientKernel kernel(h);

    std::vector<vec2f> boundary_positions;
    boundary_positions.push_back(positions[0] + vec2f(0.5f, 0.0f));
    std::vector<float> boundary_psi{2.0f};
    std::vector<int> boundary_neighbors{0};
    float rest_density = 1.0f;

    std::vector<vec2f> boundary_gradients;
    pbf::sph::detail::computeConstraintGradientsBoundary<2, GradientKernel>(
        0, rest_density, h, boundary_neighbors, positions, boundary_positions, boundary_psi,
        boundary_gradients);

    vec2f grad_i = vec2f::zero();
    float sum_grad_sq = 0.0f;
    for (const auto& gradC_j : boundary_gradients) {
        sum_grad_sq += gradC_j.dot(gradC_j);
        grad_i -= gradC_j;
    }
    sum_grad_sq += grad_i.dot(grad_i);

    vec2f gradW = kernel.deriv(positions[0] - boundary_positions[0]);
    vec2f gradC_j = -(boundary_psi[0] / rest_density) * gradW;

    float expected_sum = 2.0f * gradC_j.dot(gradC_j);
    EXPECT_NEAR(sum_grad_sq, expected_sum, 1.0e-6f);
    EXPECT_NEAR(grad_i.x, -gradC_j.x, 1.0e-6f);
    EXPECT_NEAR(grad_i.y, -gradC_j.y, 1.0e-6f);
}

TEST_F(DensityGridFixture, boundary_position_correction) {
    using GradientKernel = pbf::sph::Spikey<2>;
    GradientKernel kernel(h);

    std::vector<vec2f> boundary_positions;
    boundary_positions.push_back(positions[0] + vec2f(0.5f, 0.0f));
    std::vector<float> boundary_psi{1.25f};
    std::vector<int> boundary_neighbors{0};
    float rest_density = 1.0f;
    float lambda_i = 0.75f;

    vec2f correction = vec2f::zero();
    pbf::sph::computePositionCorrectionBoundary<2, GradientKernel>(
        0, rest_density, h, boundary_neighbors, positions, boundary_positions, boundary_psi,
        lambda_i, correction);

    vec2f gradW = kernel.deriv(positions[0] - boundary_positions[0]);
    vec2f gradC_j = -(boundary_psi[0] / rest_density) * gradW;
    vec2f expected_correction = -lambda_i * gradC_j;

    EXPECT_NEAR(correction.x, expected_correction.x, 1.0e-6f);
    EXPECT_NEAR(correction.y, expected_correction.y, 1.0e-6f);
}

TEST_F(DensityGridFixture, boundary_constraint_gradient_fd) {
    using Kernel = pbf::sph::Poly6<2>;

    std::vector<vec2f> boundary_positions;
    boundary_positions.push_back(positions[0] + vec2f(0.5f, 0.0f));
    std::vector<float> boundary_psi{1.25f};
    std::vector<int> boundary_neighbors{0};
    float rest_density = 1.0f;

    std::vector<vec2f> analytical_gradients;
    pbf::sph::detail::computeConstraintGradientsBoundary<2, Kernel>(
        0, rest_density, h, boundary_neighbors, positions, boundary_positions, boundary_psi,
        analytical_gradients);

    std::vector<vec2f> fd_gradients(boundary_neighbors.size());
    const float epsilon = 1.0e-3f;

    positions[0].x += epsilon;
    float density_plus_x = pbf::sph::computeConstraintBoundary<2, Kernel>(
        0, h, boundary_neighbors, positions, boundary_positions, boundary_psi);
    positions[0].x -= 2.0f * epsilon;
    float density_minus_x = pbf::sph::computeConstraintBoundary<2, Kernel>(
        0, h, boundary_neighbors, positions, boundary_positions, boundary_psi);
    positions[0].x += epsilon;

    positions[0].y += epsilon;
    float density_plus_y = pbf::sph::computeConstraintBoundary<2, Kernel>(
        0, h, boundary_neighbors, positions, boundary_positions, boundary_psi);
    positions[0].y -= 2.0f * epsilon;
    float density_minus_y = pbf::sph::computeConstraintBoundary<2, Kernel>(
        0, h, boundary_neighbors, positions, boundary_positions, boundary_psi);
    positions[0].y += epsilon;

    vec2f fd_grad_i = vec2f::zero();
    fd_grad_i.x = (density_plus_x - density_minus_x) / (2.0f * epsilon * rest_density);
    fd_grad_i.y = (density_plus_y - density_minus_y) / (2.0f * epsilon * rest_density);

    for (size_t k = 0; k < boundary_neighbors.size(); ++k) {
        fd_gradients[k] = -fd_grad_i;
    }

    for (size_t k = 0; k < boundary_neighbors.size(); ++k) {
        EXPECT_NEAR(analytical_gradients[k].x, fd_gradients[k].x, 1.0e-2f);
        EXPECT_NEAR(analytical_gradients[k].y, fd_gradients[k].y, 1.0e-2f);
    }
}

TEST_F(DensityGridFixture, constraint_gradient) {
    using Kernel = pbf::sph::Poly6<2>;  // Same kernel as computeConstraint test

    float mass = 1.2f;
    float rest_density = 0.95f;
    std::vector<int> neighbors;

    // Get neighbors for particle 0 (self-particle)
    neighbors = testing::get_neighbors_slow<2>(0, positions, h);

    // Compute analytical gradients
    std::vector<vec2f> analytical_gradients;
    pbf::sph::detail::computeConstraintGradients<2, Kernel>(
        0, rest_density, mass, h, neighbors, positions, analytical_gradients);

    // Compute finite difference gradients
    std::vector<vec2f> fd_gradients;
    fd_gradients.resize(neighbors.size() + 1);

    // Use smaller perturbation for better accuracy
    const float epsilon = 1.0e-3f;

    // Compute finite difference for self-gradient (index 0)
    // Perturb in +x direction
    positions[0].x += epsilon;
    float c_plus_x = pbf::sph::computeConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
    positions[0].x -= 2 * epsilon;
    float c_minus_x = pbf::sph::computeConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
    positions[0].x += epsilon;  // Restore

    // Perturb in +y direction
    positions[0].y += epsilon;
    float c_plus_y = pbf::sph::computeConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
    positions[0].y -= 2 * epsilon;
    float c_minus_y = pbf::sph::computeConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
    positions[0].y += epsilon;  // Restore

    fd_gradients[0].x = (c_plus_x - c_minus_x) / (2 * epsilon);
    fd_gradients[0].y = (c_plus_y - c_minus_y) / (2 * epsilon);

    // Compute finite difference for each neighbor gradient
    for (size_t k = 0; k < neighbors.size(); ++k) {
        int neighbor_index = neighbors[k];
        // Perturb neighbor in +x direction
        positions[neighbor_index].x += epsilon;
        c_plus_x = pbf::sph::computeConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
        positions[neighbor_index].x -= 2 * epsilon;
        c_minus_x = pbf::sph::computeConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
        positions[neighbor_index].x += epsilon;  // Restore

        // Perturb neighbor in +y direction
        positions[neighbor_index].y += epsilon;
        c_plus_y = pbf::sph::computeConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
        positions[neighbor_index].y -= 2 * epsilon;
        c_minus_y = pbf::sph::computeConstraint<2, Kernel>(0, rest_density, mass, h, neighbors, positions);
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

TEST_F(DensityGridFixture, lambda_clamps_negative_constraint) {
    using ConstraintKernel = pbf::sph::Poly6<2>;
    using GradientKernel = pbf::sph::Spikey<2>;

    float mass = 1.2f;
    float epsilon = 1.0e-6f;

    std::vector<int> neighbors = testing::get_neighbors_slow<2>(0, positions, h);
    float density = pbf::sph::computeDensity<2, ConstraintKernel>(0, mass, h, neighbors, positions);
    float rest_density = density * 1.1f;

    float constraint = pbf::sph::computeConstraint<2, ConstraintKernel>(
        0, rest_density, mass, h, neighbors, positions);
    EXPECT_LT(constraint, 0.0f);

    std::vector<vec2f> gradients;
    pbf::sph::detail::computeConstraintGradients<2, GradientKernel>(
        0, rest_density, mass, h, neighbors, positions, gradients);

    std::vector<int> boundary_neighbors;
    std::vector<vec2f> boundary_positions;
    std::vector<float> boundary_psi;

    float sum_grad_sq = 0.0f;
    for (const auto& grad : gradients) {
        sum_grad_sq += grad.dot(grad);
    }
    float expected_unclamped = -constraint / (sum_grad_sq + epsilon);
    // Negative constraint would produce a positive lambda, but we clamp to zero.
    EXPECT_GT(expected_unclamped, 0.0f);

    float lambda = pbf::sph::computeLambdaWithBoundary<2, ConstraintKernel, GradientKernel>(
        0, rest_density, mass, h, epsilon, neighbors, boundary_neighbors,
        positions, boundary_positions, boundary_psi);
    EXPECT_NEAR(lambda, 0.0f, 1.0e-6f);
}

TEST_F(DensityGridFixture, lambda_keeps_positive_constraint) {
    using ConstraintKernel = pbf::sph::Poly6<2>;
    using GradientKernel = pbf::sph::Spikey<2>;

    float mass = 1.2f;
    float epsilon = 1.0e-6f;

    std::vector<int> neighbors = testing::get_neighbors_slow<2>(0, positions, h);
    float density = pbf::sph::computeDensity<2, ConstraintKernel>(0, mass, h, neighbors, positions);
    float rest_density = density * 0.5f;

    float constraint = pbf::sph::computeConstraint<2, ConstraintKernel>(
        0, rest_density, mass, h, neighbors, positions);
    EXPECT_GT(constraint, 0.0f);

    std::vector<vec2f> gradients;
    pbf::sph::detail::computeConstraintGradients<2, GradientKernel>(
        0, rest_density, mass, h, neighbors, positions, gradients);

    std::vector<int> boundary_neighbors;
    std::vector<vec2f> boundary_positions;
    std::vector<float> boundary_psi;

    float sum_grad_sq = 0.0f;
    for (const auto& grad : gradients) {
        sum_grad_sq += grad.dot(grad);
    }
    // Positive constraint should keep the negative lambda unclamped.
    float expected_lambda = -constraint / (sum_grad_sq + epsilon);

    float lambda = pbf::sph::computeLambdaWithBoundary<2, ConstraintKernel, GradientKernel>(
        0, rest_density, mass, h, epsilon, neighbors, boundary_neighbors,
        positions, boundary_positions, boundary_psi);
    EXPECT_LT(lambda, 0.0f);
    EXPECT_NEAR(lambda, expected_lambda, 1.0e-6f);
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
        pbf::sph::detail::computeConstraintGradients<2, Kernel>(
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

using testing::DensityGridFixture3D;
TEST_F(DensityGridFixture3D, density_constraint_3d)
{
    float mass = 1.2f;
    std::vector<int> neighbors;
    float rest_density = 0.95f;
    float density = pbf::sph::computeDensity<3, Kernel>(0, mass, h, neighbors, positions);
    float density_constraint = pbf::sph::computeConstraint<3, Kernel>(0, rest_density, mass, h, neighbors, positions);
    float tol = 1.0e-4f;
    EXPECT_NEAR(density_constraint, density / rest_density - 1.0f, tol);
}

TEST_F(DensityGridFixture3D, constraint_gradient_3d) {
    using Kernel = pbf::sph::Poly6<3>;

    float mass = 1.2f;
    float rest_density = 0.95f;
    std::vector<int> neighbors;

    neighbors = testing::get_neighbors_slow<3>(0, positions, h);

    std::vector<pbf::vec3f> analytical_gradients;
    pbf::sph::detail::computeConstraintGradients<3, Kernel>(
        0, rest_density, mass, h, neighbors, positions, analytical_gradients);

    std::vector<pbf::vec3f> fd_gradients;
    fd_gradients.resize(neighbors.size() + 1);

    const float epsilon = 1.0e-3f;

    positions[0].x += epsilon;
    float c_plus_x = pbf::sph::computeConstraint<3, Kernel>(0, rest_density, mass, h, neighbors, positions);
    positions[0].x -= 2 * epsilon;
    float c_minus_x = pbf::sph::computeConstraint<3, Kernel>(0, rest_density, mass, h, neighbors, positions);
    positions[0].x += epsilon;

    positions[0].y += epsilon;
    float c_plus_y = pbf::sph::computeConstraint<3, Kernel>(0, rest_density, mass, h, neighbors, positions);
    positions[0].y -= 2 * epsilon;
    float c_minus_y = pbf::sph::computeConstraint<3, Kernel>(0, rest_density, mass, h, neighbors, positions);
    positions[0].y += epsilon;

    positions[0].z += epsilon;
    float c_plus_z = pbf::sph::computeConstraint<3, Kernel>(0, rest_density, mass, h, neighbors, positions);
    positions[0].z -= 2 * epsilon;
    float c_minus_z = pbf::sph::computeConstraint<3, Kernel>(0, rest_density, mass, h, neighbors, positions);
    positions[0].z += epsilon;

    fd_gradients[0].x = (c_plus_x - c_minus_x) / (2 * epsilon);
    fd_gradients[0].y = (c_plus_y - c_minus_y) / (2 * epsilon);
    fd_gradients[0].z = (c_plus_z - c_minus_z) / (2 * epsilon);

    for (size_t k = 0; k < neighbors.size(); ++k) {
        int neighbor_index = neighbors[k];
        positions[neighbor_index].x += epsilon;
        c_plus_x = pbf::sph::computeConstraint<3, Kernel>(0, rest_density, mass, h, neighbors, positions);
        positions[neighbor_index].x -= 2 * epsilon;
        c_minus_x = pbf::sph::computeConstraint<3, Kernel>(0, rest_density, mass, h, neighbors, positions);
        positions[neighbor_index].x += epsilon;

        positions[neighbor_index].y += epsilon;
        c_plus_y = pbf::sph::computeConstraint<3, Kernel>(0, rest_density, mass, h, neighbors, positions);
        positions[neighbor_index].y -= 2 * epsilon;
        c_minus_y = pbf::sph::computeConstraint<3, Kernel>(0, rest_density, mass, h, neighbors, positions);
        positions[neighbor_index].y += epsilon;

        positions[neighbor_index].z += epsilon;
        c_plus_z = pbf::sph::computeConstraint<3, Kernel>(0, rest_density, mass, h, neighbors, positions);
        positions[neighbor_index].z -= 2 * epsilon;
        c_minus_z = pbf::sph::computeConstraint<3, Kernel>(0, rest_density, mass, h, neighbors, positions);
        positions[neighbor_index].z += epsilon;

        fd_gradients[k + 1].x = (c_plus_x - c_minus_x) / (2 * epsilon);
        fd_gradients[k + 1].y = (c_plus_y - c_minus_y) / (2 * epsilon);
        fd_gradients[k + 1].z = (c_plus_z - c_minus_z) / (2 * epsilon);
    }

    EXPECT_NEAR(analytical_gradients[0].x, fd_gradients[0].x, 1.0e-2f);
    EXPECT_NEAR(analytical_gradients[0].y, fd_gradients[0].y, 1.0e-2f);
    EXPECT_NEAR(analytical_gradients[0].z, fd_gradients[0].z, 1.0e-2f);

    for (size_t k = 0; k < neighbors.size(); ++k) {
        EXPECT_NEAR(analytical_gradients[k + 1].x, fd_gradients[k + 1].x, 1.0e-2f);
        EXPECT_NEAR(analytical_gradients[k + 1].y, fd_gradients[k + 1].y, 1.0e-2f);
        EXPECT_NEAR(analytical_gradients[k + 1].z, fd_gradients[k + 1].z, 1.0e-2f);
    }
}

TEST_F(DensityGridFixture3D, lambda_clamps_negative_constraint_3d) {
    using ConstraintKernel = pbf::sph::Poly6<3>;
    using GradientKernel = pbf::sph::Spikey<3>;

    float mass = 1.2f;
    float epsilon = 1.0e-6f;

    std::vector<int> neighbors = testing::get_neighbors_slow<3>(0, positions, h);
    float density = pbf::sph::computeDensity<3, ConstraintKernel>(0, mass, h, neighbors, positions);
    float rest_density = density * 1.1f;

    float constraint = pbf::sph::computeConstraint<3, ConstraintKernel>(
        0, rest_density, mass, h, neighbors, positions);
    EXPECT_LT(constraint, 0.0f);

    std::vector<pbf::vec3f> gradients;
    pbf::sph::detail::computeConstraintGradients<3, GradientKernel>(
        0, rest_density, mass, h, neighbors, positions, gradients);

    std::vector<int> boundary_neighbors;
    std::vector<pbf::vec3f> boundary_positions;
    std::vector<float> boundary_psi;

    float sum_grad_sq = 0.0f;
    for (const auto& grad : gradients) {
        sum_grad_sq += grad.dot(grad);
    }
    float expected_unclamped = -constraint / (sum_grad_sq + epsilon);
    EXPECT_GT(expected_unclamped, 0.0f);

    float lambda = pbf::sph::computeLambdaWithBoundary<3, ConstraintKernel, GradientKernel>(
        0, rest_density, mass, h, epsilon, neighbors, boundary_neighbors,
        positions, boundary_positions, boundary_psi);
    EXPECT_NEAR(lambda, 0.0f, 1.0e-6f);
}

TEST_F(DensityGridFixture3D, lambda_keeps_positive_constraint_3d) {
    using ConstraintKernel = pbf::sph::Poly6<3>;
    using GradientKernel = pbf::sph::Spikey<3>;

    float mass = 1.2f;
    float epsilon = 1.0e-6f;

    std::vector<int> neighbors = testing::get_neighbors_slow<3>(0, positions, h);
    float density = pbf::sph::computeDensity<3, ConstraintKernel>(0, mass, h, neighbors, positions);
    float rest_density = density * 0.5f;

    float constraint = pbf::sph::computeConstraint<3, ConstraintKernel>(
        0, rest_density, mass, h, neighbors, positions);
    EXPECT_GT(constraint, 0.0f);

    std::vector<pbf::vec3f> gradients;
    pbf::sph::detail::computeConstraintGradients<3, GradientKernel>(
        0, rest_density, mass, h, neighbors, positions, gradients);

    std::vector<int> boundary_neighbors;
    std::vector<pbf::vec3f> boundary_positions;
    std::vector<float> boundary_psi;

    float sum_grad_sq = 0.0f;
    for (const auto& grad : gradients) {
        sum_grad_sq += grad.dot(grad);
    }
    float expected_lambda = -constraint / (sum_grad_sq + epsilon);

    float lambda = pbf::sph::computeLambdaWithBoundary<3, ConstraintKernel, GradientKernel>(
        0, rest_density, mass, h, epsilon, neighbors, boundary_neighbors,
        positions, boundary_positions, boundary_psi);
    EXPECT_LT(lambda, 0.0f);
    EXPECT_NEAR(lambda, expected_lambda, 1.0e-6f);
}

TEST_F(DensityGridFixture3D, position_correction_matrix_vector_test_3d) {
    using Kernel = pbf::sph::Spikey<3>;

    float mass = 1.2f;
    float rest_density = 0.95f;
    float epsilon = 1.0e-6f;

    std::vector<float> lambdas(nparticles);
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < nparticles; ++i) {
        lambdas[i] = dis(gen);
    }

    std::vector<std::vector<float>> gradc(nparticles * 3, std::vector<float>(nparticles, 0.0f));

    for (int i = 0; i < nparticles; ++i) {
        std::vector<int> neighbors = testing::get_neighbors_slow<3>(i, positions, h);
        std::vector<pbf::vec3f> gradients;
        pbf::sph::detail::computeConstraintGradients<3, Kernel>(
            i, rest_density, mass, h, neighbors, positions, gradients);

        gradc[i * 3][i] = gradients[0].x;
        gradc[i * 3 + 1][i] = gradients[0].y;
        gradc[i * 3 + 2][i] = gradients[0].z;

        for (size_t k = 0; k < neighbors.size(); ++k) {
            int neighbor_index = neighbors[k];
            gradc[neighbor_index * 3][i] = gradients[k + 1].x;
            gradc[neighbor_index * 3 + 1][i] = gradients[k + 1].y;
            gradc[neighbor_index * 3 + 2][i] = gradients[k + 1].z;
        }
    }

    std::vector<pbf::vec3f> deltap(nparticles);
    for (int i = 0; i < nparticles; ++i) {
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float sum_z = 0.0f;
        for (int j = 0; j < nparticles; ++j) {
            sum_x += gradc[i * 3][j] * lambdas[j];
            sum_y += gradc[i * 3 + 1][j] * lambdas[j];
            sum_z += gradc[i * 3 + 2][j] * lambdas[j];
        }
        deltap[i] = pbf::vec3f(sum_x, sum_y, sum_z);
    }

    for (int i = 0; i < nparticles; ++i) {
        std::vector<int> neighbors = testing::get_neighbors_slow<3>(i, positions, h);
        pbf::vec3f correction;
        pbf::sph::calculatePositionCorrection<3, Kernel>(
            i, rest_density, mass, h, neighbors, lambdas, positions, correction);

        EXPECT_NEAR(correction.x, deltap[i].x, 1.0e-4f) << "Particle " << i;
        EXPECT_NEAR(correction.y, deltap[i].y, 1.0e-4f) << "Particle " << i;
        EXPECT_NEAR(correction.z, deltap[i].z, 1.0e-4f) << "Particle " << i;
    }
}
