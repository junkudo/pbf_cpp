#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>

#include "pbf/pbf_kernels.h"
#include "pbf/spatial_hash.h"
#include "pbf/sph_kernels.h"
#include "pbf/vec2f.h"
#include "pbf/visualization.h"
#include "raylib.h"

using namespace pbf;

// Configuration constants
constexpr float PARTICLE_SPACING = 0.01f;
constexpr float KERNEL_RADIUS = 2.5f * PARTICLE_SPACING;
constexpr float REST_DENSITY = 1.0f;
constexpr float GRAVITY = 9.8f;
constexpr float TIME_STEP = 1.0f / 240.0f;
constexpr float MASS = REST_DENSITY * PARTICLE_SPACING * PARTICLE_SPACING;
constexpr int NUM_ITERATIONS = 50;
constexpr float JITTER_FACTOR = 0.05f;

// PBF solver parameters
constexpr float CONSTRAINT_EPSILON = 1.0e-6f;
constexpr float POSITION_CORRECTION_FACTOR = 0.1f;

class ParticleSystem {
private:
    std::vector<vec2f> positions_;
    std::vector<vec2f> velocities_;
    std::vector<float> lambdas_;
    int num_particles_;

public:
    ParticleSystem(int num_x, int num_y) : num_particles_(num_x * num_y) {
        positions_.resize(num_particles_);
        velocities_.resize(num_particles_);
        lambdas_.resize(num_particles_);

        // Initialize particles in a jittered grid
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> jitter(-JITTER_FACTOR * PARTICLE_SPACING, JITTER_FACTOR * PARTICLE_SPACING);

        int index = 0;
        for (int y = 0; y < num_y; ++y) {
            for (int x = 0; x < num_x; ++x) {
                float px = x * PARTICLE_SPACING + jitter(gen);
                float py = y * PARTICLE_SPACING + jitter(gen) + 0.05f; // Offset from origin
                positions_[index] = vec2f(px, py);
                velocities_[index] = vec2f(0.0f, 0.0f);
                ++index;
            }
        }
    }

    void updateVelocityFromGravity() {
        for (auto& v : velocities_) {
            v.y -= GRAVITY * TIME_STEP;
        }
    }

    std::vector<vec2f> predictPositions() const {
        std::vector<vec2f> predicted = positions_;
        for (int i = 0; i < num_particles_; ++i) {
            predicted[i] += velocities_[i] * TIME_STEP;
        }
        return predicted;
    }

    void updatePositions(const std::vector<vec2f>& corrections) {
        for (int i = 0; i < num_particles_; ++i) {
            positions_[i] += corrections[i];
            velocities_[i] = corrections[i] / TIME_STEP;
        }
    }

    const std::vector<vec2f>& getPositions() const { return positions_; }
    int getNumParticles() const { return num_particles_; }
};

float calculateLambda(int i, const std::vector<int>& neighbors,
                      const std::vector<vec2f>& positions) {
    return sph::computeLambda<2, sph::Poly6<2>, sph::Spikey<2>>(i, REST_DENSITY, MASS,
                                                                KERNEL_RADIUS, CONSTRAINT_EPSILON,
                                                                neighbors, positions);
}

std::vector<vec2f> solveConstraints(const std::vector<vec2f>& predicted_positions,
                                   const std::vector<std::vector<int>>& neighbors,
                                   const std::vector<float>& lambdas) {
    std::vector<vec2f> corrections(neighbors.size(), vec2f(0.0f, 0.0f));

    // Calculate position corrections
    for (int i = 0; i < neighbors.size(); ++i) {
        sph::calculatePositionCorrection<2, sph::Spikey<2>>(i, REST_DENSITY, MASS, KERNEL_RADIUS,
                                                             neighbors[i], lambdas, predicted_positions,
                                                             corrections[i]);
    }

    return corrections;
}

void runSimulationStep(ParticleSystem& system, SpatialHash& spatial_hash) {
    // 1. Update velocities from gravity
    system.updateVelocityFromGravity();

    // 2. Predict positions
    auto predicted_positions = system.predictPositions();

    // 3. Build spatial hash with predicted positions
    spatial_hash.update(predicted_positions);

    // 4. Find neighbors for each particle
    auto neighbors = spatial_hash.getAllNeighbors();

    // 5. Solve constraints (50 iterations)
    std::vector<vec2f> total_corrections(system.getNumParticles(), vec2f(0.0f, 0.0f));
    std::vector<float> lambdas(system.getNumParticles(), 0.0f);

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // Calculate lambdas
        for (int i = 0; i < system.getNumParticles(); ++i) {
            lambdas[i] = calculateLambda(i, neighbors[i], predicted_positions);
        }

        auto corrections = solveConstraints(predicted_positions, neighbors, lambdas);

        // Apply under-relaxation
        for (int i = 0; i < system.getNumParticles(); ++i) {
            total_corrections[i] += POSITION_CORRECTION_FACTOR * corrections[i];
            predicted_positions[i] += POSITION_CORRECTION_FACTOR * corrections[i];
        }
    }

    // 6. Update velocities from position corrections
    // 7. Update positions to final positions
    system.updatePositions(total_corrections);
}


int main() {
    std::cout << "PBF Fluid Simulation with Visualization" << std::endl;
    std::cout << "Particles: " << (10 * 10) << std::endl;
    std::cout << "Kernel radius: " << KERNEL_RADIUS << std::endl;
    std::cout << "Time step: " << TIME_STEP << std::endl;
    std::cout << "Solver iterations: " << NUM_ITERATIONS << std::endl;

    // Create visualization configuration
    pbf::visualization::VisualizationConfig config;
    config.screenWidth = 800;
    config.screenHeight = 600;
    config.scale = 4000.0f;
    config.offsetX = 100.0f;   // Move origin 100 pixels from left edge
    config.offsetY = 500.0f;   // Move origin 500 pixels from top (closer to bottom)
    config.particleRadius = 3.0f;
    config.originDotRadius = 3.0f;
    config.labelFontSize = 10;
    config.timeFontSize = 20;

    // Initialize raylib
    InitWindow(config.screenWidth, config.screenHeight, "PBF Fluid Simulation");

    ParticleSystem system(10, 10);

    // Build spatial hash
    float domain_size = 10.0f * PARTICLE_SPACING;
    SpatialHash spatial_hash(-domain_size, -domain_size, domain_size, domain_size, KERNEL_RADIUS);

    float simulationTime = 0.0f;  // Add time tracking

    // Simple simulation + display loop
    while (!WindowShouldClose()) {
        // Run one simulation step
        runSimulationStep(system, spatial_hash);
        simulationTime += TIME_STEP;  // Update simulation time

        // Display immediately after simulation
        BeginDrawing();
        ClearBackground(BLACK);

        // Draw origin visualization
        pbf::visualization::drawOrigin(config);

        // Draw particles and their influence circles
        pbf::visualization::drawParticles(system.getPositions(), KERNEL_RADIUS, config);

        // Draw time display
        pbf::visualization::drawTime(simulationTime, config);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
