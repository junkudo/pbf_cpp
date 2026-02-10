#include <array>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

#include "pbf/pbf_kernels.h"
#include "pbf/spatial_hash.h"
#include "pbf/sph_kernels.h"
#include "pbf/particle_system.h"
#include "pbf/vec2f.h"
#include "pbf/visualization.h"
#include "pbf/config.h"
#include "raylib.h"

using namespace pbf;

float calculateLambda(int i, const std::vector<int>& neighbors,
                      const std::vector<vec2f>& positions, const PhysicsConfig& config) {
    return sph::computeLambda<2, sph::Poly6<2>, sph::Spikey<2>>(i, config.restDensity, config.mass,
                                                                config.kernelRadius, config.constraintEpsilon,
                                                                neighbors, positions);
}

std::vector<vec2f> solveConstraints(const std::vector<vec2f>& predicted_positions,
                                   const std::vector<std::vector<int>>& neighbors,
                                   const std::vector<float>& lambdas, const PhysicsConfig& config) {
    std::vector<vec2f> corrections(neighbors.size(), vec2f(0.0f, 0.0f));

    // Calculate position corrections
    for (int i = 0; i < neighbors.size(); ++i) {
        sph::calculatePositionCorrection<2, sph::Spikey<2>>(i, config.restDensity, config.mass, config.kernelRadius,
                                                             neighbors[i], lambdas, predicted_positions,
                                                             corrections[i]);
    }

    return corrections;
}

void runSimulationStep(ParticleSystem<2>& system, SpatialHash<2>& spatial_hash, const SolverConfig& solver_config) {
    const PhysicsConfig& physics_config = system.config_;  // Get physics config from system

    // 1. Update velocities from gravity
    system.updateVelocityFromGravity();

    // 2. Predict positions
    auto predicted_positions = system.predictPositions();

    // 3. Build spatial hash with predicted positions
    spatial_hash.update(predicted_positions);

    // 4. Find neighbors for each particle
    auto neighbors = spatial_hash.getAllNeighbors();

    // 5. Solve constraints (iterations from solver config)
    std::vector<vec2f> total_corrections(system.getNumParticles(), vec2f::zero());
    std::vector<float> lambdas(system.getNumParticles(), 0.0f);

    for (int iter = 0; iter < solver_config.numIterations; ++iter) {
        // Calculate lambdas
        for (int i = 0; i < system.getNumParticles(); ++i) {
            lambdas[i] = calculateLambda(i, neighbors[i], predicted_positions, physics_config);
        }

        auto corrections = solveConstraints(predicted_positions, neighbors, lambdas, physics_config);

        // Apply under-relaxation
        for (int i = 0; i < system.getNumParticles(); ++i) {
            total_corrections[i] += solver_config.positionCorrectionFactor * corrections[i];
            predicted_positions[i] += solver_config.positionCorrectionFactor * corrections[i];
        }
    }

    // 6. Update velocities from position corrections
    // 7. Update positions to final positions
    system.updatePositions(total_corrections);
}


int main() {
    // Create complete application configuration
    AppConfig app_config;

    // Set appropriate particle radius to make particles visible but not too large
    app_config.visualization.particleRadius = 4.0f;

    std::cout << "PBF Fluid Simulation with Visualization" << std::endl;
    std::cout << "Particles: " << (10 * 10) << std::endl;
    std::cout << "Kernel radius: " << app_config.physics.kernelRadius << std::endl;
    std::cout << "Time step: " << app_config.physics.timeStep << std::endl;
    std::cout << "Solver iterations: " << app_config.solver.numIterations << std::endl;

    // Initialize raylib
    InitWindow(app_config.visualization.screenWidth, app_config.visualization.screenHeight, "PBF Fluid Simulation");

    const std::array<int, 2> particle_counts{10, 10};
    const vec2f origin_offset(0.0f, 0.05f);
    ParticleSystem<2> system(particle_counts, app_config.physics, origin_offset);

    // Build spatial hash
    float domain_size = 10.0f * app_config.physics.particleSpacing;
    vec2f min_bounds(-domain_size, -domain_size);
    vec2f max_bounds(domain_size, domain_size);
    SpatialHash<2> spatial_hash(min_bounds, max_bounds, app_config.physics.kernelRadius);

    float simulationTime = 0.0f;  // Add time tracking

    // Simple simulation + display loop
    while (!WindowShouldClose()) {
        // Run one simulation step
        runSimulationStep(system, spatial_hash, app_config.solver);
        simulationTime += app_config.physics.timeStep;  // Update simulation time

        // Display immediately after simulation
        BeginDrawing();
        ClearBackground(BLACK);

        // Draw origin visualization
        pbf::visualization::drawOrigin(app_config.visualization);

        // Draw particles and their influence circles
        pbf::visualization::drawParticles(system.positions_, app_config.physics.kernelRadius, app_config.visualization);

        // Draw time display
        pbf::visualization::drawTime(simulationTime, app_config.visualization);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
