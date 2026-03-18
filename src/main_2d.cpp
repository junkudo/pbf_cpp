#include <array>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <chrono>

#include "pbf/pbf_kernels.h"
#include "pbf/spatial_hash.h"
#include "pbf/sph_kernels.h"
#include "pbf/particle_system.h"
#include "pbf/vec2f.h"
#include "pbf/visualization.h"
#include "pbf/config.h"
#include "raylib.h"

using namespace pbf;

namespace {

// --- Tunable constants for the 2D demo ---
constexpr float kParticleSpacing = 0.005f; // Simulation spacing in meters.
constexpr float kVisualizationScale = 800.0f; // Pixels per meter in render space.
constexpr float kOriginOffsetScaleX = 0.5f; // Center the grid horizontally.
constexpr float kOriginOffsetScaleY = 0.9f; // Move the grid toward the bottom of the screen.
constexpr float kOriginOffsetY = 0.05f; // Initial vertical offset in simulation space.
constexpr float kDomainSizeScale = 10.0f; // Domain size in multiples of particle spacing.
constexpr float kContainerWidthScale = 3.0f; // Container width in multiples of domain size.
constexpr float kContainerHeightScale = 2.0f; // Container height in multiples of domain size.
constexpr float kBoundaryPressureStrength = 1.2f; // Boundary pressure strength multiplier.
constexpr int kTargetFps = 60; // Target frame rate.
constexpr int kParticleCountX = 20; // Particles per row.
constexpr int kParticleCountY = 20; // Particles per column.
constexpr int kSolverIterations = 15; // Solver iterations per step.

// Overlay layout constants.
constexpr int kOverlayX = 10;
constexpr int kOverlayY = 10;
constexpr int kOverlayFontSize = 18;
constexpr int kOverlayLineSpacing = 20;

struct StepTimings {
    double total_ms = 0.0;
    double gravity_ms = 0.0;
    double predict_ms = 0.0;
    double hash_update_ms = 0.0;
    double neighbor_ms = 0.0;
    double lambda_ms = 0.0;
    double correction_ms = 0.0;
    double apply_corrections_ms = 0.0;
    double update_positions_ms = 0.0;
    double xsph_ms = 0.0;
    double render_ms = 0.0;
    double render_begin_ms = 0.0;
    double render_sim_ms = 0.0;
    double render_overlay_ms = 0.0;
    double render_end_ms = 0.0;
};

// Draw the CPU timing overlay for a simulation step.
void drawOverlay(const StepTimings& timings) {
    const int overlay_x = kOverlayX;
    const int overlay_y = kOverlayY;
    const int overlay_size = kOverlayFontSize;
    const int overlay_line = kOverlayLineSpacing;
    int line_index = 0;
    DrawText(TextFormat("Step total: %.2f ms", timings.total_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("  Gravity: %.2f ms", timings.gravity_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("  Predict: %.2f ms", timings.predict_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("  Hash update: %.2f ms", timings.hash_update_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("  Neighbors: %.2f ms", timings.neighbor_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("  Constraints: %.2f ms", timings.lambda_ms + timings.correction_ms
                                                  + timings.apply_corrections_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("    Lambda: %.2f ms", timings.lambda_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("    Corrections: %.2f ms", timings.correction_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("    Apply: %.2f ms", timings.apply_corrections_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("  Update positions: %.2f ms", timings.update_positions_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("  XSPH: %.2f ms", timings.xsph_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("Render (CPU): %.2f ms", timings.render_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("FPS: %d", GetFPS()),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("  Begin/Clear: %.2f ms", timings.render_begin_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("  Sim draw: %.2f ms", timings.render_sim_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("  Overlay text: %.2f ms", timings.render_overlay_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
    DrawText(TextFormat("  EndDrawing: %.2f ms", timings.render_end_ms),
             overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
}

void addWall(const vec2f& min_corner, const vec2f& max_corner,
             float particle_distance,
             std::vector<vec2f>& boundary_positions) {
    const vec2f diff = max_corner - min_corner;
    const int steps_x = static_cast<int>(diff.x / particle_distance) + 1;
    const int steps_y = static_cast<int>(diff.y / particle_distance) + 1;

    const int start_index = static_cast<int>(boundary_positions.size());
    boundary_positions.resize(start_index + steps_x * steps_y, vec2f::zero());

    for (int i = 0; i < steps_x; ++i) {
        for (int j = 0; j < steps_y; ++j) {
            const vec2f curr_pos = min_corner
                                   + vec2f(static_cast<float>(i), static_cast<float>(j))
                                       * particle_distance;
            boundary_positions[start_index + i * steps_y + j] = curr_pos;
        }
    }
}

std::vector<vec2f> initContainerBoundary(float container_width,
                                         float container_height,
                                         float particle_distance) {
    const float x1 = -0.5f * container_width;
    const float x2 = 0.5f * container_width;
    const float y1 = 0.0f;
    const float y2 = container_height;

    std::vector<vec2f> boundary_positions;
    addWall(vec2f(x1, y1), vec2f(x2, y1), particle_distance, boundary_positions);
    addWall(vec2f(x1, y1), vec2f(x1, y2), particle_distance, boundary_positions);
    addWall(vec2f(x2, y1), vec2f(x2, y2), particle_distance, boundary_positions);

    const float inner_x1 = x1 + particle_distance;
    const float inner_x2 = x2 - particle_distance;
    const float inner_y1 = y1 + particle_distance;
    const float inner_y2 = y2 - particle_distance;
    addWall(vec2f(inner_x1, inner_y1), vec2f(inner_x2, inner_y1), particle_distance, boundary_positions);
    addWall(vec2f(inner_x1, inner_y1), vec2f(inner_x1, inner_y2), particle_distance, boundary_positions);
    addWall(vec2f(inner_x2, inner_y1), vec2f(inner_x2, inner_y2), particle_distance, boundary_positions);
    return boundary_positions;
}

// Build boundary particles, hash, and psi values for the container walls.
void initBoundaryData(float container_width,
                      float container_height,
                      float boundary_spacing,
                      float kernel_radius,
                      float rest_density,
                      std::vector<vec2f>& boundary_positions,
                      std::vector<float>& boundary_psi,
                      SpatialHash<2>& boundary_hash) {
    boundary_positions = initContainerBoundary(container_width, container_height, boundary_spacing);
    boundary_psi.assign(boundary_positions.size(), 0.0f);
    boundary_hash.update(boundary_positions);

    std::vector<int> boundary_neighbors;
    for (size_t i = 0; i < boundary_psi.size(); ++i) {
        const vec2f& pi = boundary_positions[i];
        boundary_hash.getNeighborsForPosition(pi, boundary_neighbors);
        boundary_psi[i] = sph::computeBoundaryPsi<2, sph::CubicSpline<2>>(
            static_cast<int>(i),
            kernel_radius,
            rest_density,
            kBoundaryPressureStrength,
            boundary_neighbors,
            boundary_positions);
    }
}

// Initialize 2D demo config defaults.
AppConfig initAppConfig2D() {
    AppConfig app_config;
    app_config.visualization.scale = kVisualizationScale;
    app_config.visualization.offsetX = kOriginOffsetScaleX
                                       * static_cast<float>(app_config.visualization.screenWidth);
    app_config.visualization.offsetY = kOriginOffsetScaleY
                                       * static_cast<float>(app_config.visualization.screenHeight);
    app_config.physics.particleSpacing = kParticleSpacing;
    app_config.physics.calculateDerivedValues2D();
    // Render radius in pixels derived from kernel radius.
    app_config.visualization.particleRadius = app_config.physics.kernelRadius
                                              * app_config.visualization.scale;
    // Override default solver iterations.
    app_config.solver.numIterations = kSolverIterations;
    return app_config;
}

// Render the current frame and record timing breakdowns.
void renderFrame(const AppConfig& app_config,
                 const vec2f& min_bounds,
                 const vec2f& max_bounds,
                 const std::vector<vec2f>& positions,
                 float kernel_radius,
                 float simulation_time,
                 StepTimings& timings) {
    const auto render_start = std::chrono::high_resolution_clock::now();
    const auto render_begin_start = std::chrono::high_resolution_clock::now();
    BeginDrawing();
    ClearBackground(BLACK);
    const auto render_begin_end = std::chrono::high_resolution_clock::now();
    timings.render_begin_ms = std::chrono::duration<double, std::milli>(
        render_begin_end - render_begin_start).count();

    const auto render_sim_start = std::chrono::high_resolution_clock::now();
    pbf::visualization::drawOrigin(app_config.visualization);
    pbf::visualization::drawWalls(min_bounds, max_bounds, app_config.visualization);
    pbf::visualization::drawParticles(positions, kernel_radius, app_config.visualization);
    pbf::visualization::drawTime(simulation_time, app_config.visualization);
    const auto render_sim_end = std::chrono::high_resolution_clock::now();
    timings.render_sim_ms = std::chrono::duration<double, std::milli>(
        render_sim_end - render_sim_start).count();

    const auto render_overlay_start = std::chrono::high_resolution_clock::now();
    drawOverlay(timings);
    const auto render_overlay_end = std::chrono::high_resolution_clock::now();
    timings.render_overlay_ms = std::chrono::duration<double, std::milli>(
        render_overlay_end - render_overlay_start).count();

    const auto render_end_start = std::chrono::high_resolution_clock::now();
    EndDrawing();
    const auto render_end = std::chrono::high_resolution_clock::now();
    timings.render_end_ms = std::chrono::duration<double, std::milli>(
        render_end - render_end_start).count();
    timings.render_ms = std::chrono::duration<double, std::milli>(
        render_end - render_start).count();
}

void solveConstraints(const std::vector<vec2f>& predicted_positions,
                      const std::vector<std::vector<int>>& neighbors,
                      const std::vector<std::vector<int>>& boundary_neighbors,
                      const std::vector<vec2f>& boundary_positions,
                      const std::vector<float>& boundary_psi,
                      const std::vector<float>& lambdas,
                      const PhysicsConfig& config,
                      std::vector<vec2f>& corrections) {
    corrections.clear();
    corrections.resize(neighbors.size(), vec2f::zero());

    for (int i = 0; i < static_cast<int>(neighbors.size()); ++i) {
        sph::calculatePositionCorrection<2, sph::CubicSpline<2>>(
            i, config.restDensity, config.mass, config.kernelRadius,
            neighbors[i], lambdas, predicted_positions,
            corrections[i]);
        sph::computePositionCorrectionBoundary<2, sph::CubicSpline<2>>(
            i, config.restDensity, config.kernelRadius,
            boundary_neighbors[i], predicted_positions, boundary_positions,
            boundary_psi, lambdas[i], corrections[i]);
    }
}

void runSimulationStep(ParticleSystem<2>& system, SpatialHash<2>& spatial_hash,
                       const SpatialHash<2>& boundary_hash,
                       const std::vector<vec2f>& boundary_positions,
                       const std::vector<float>& boundary_psi,
                       const SolverConfig& solver_config,
                       std::vector<vec2f>& predicted_positions,
                       std::vector<std::vector<int>>& neighbors,
                       std::vector<std::vector<int>>& boundary_neighbors,
                       std::vector<vec2f>& total_corrections,
                       std::vector<vec2f>& corrections,
                       std::vector<float>& lambdas,
                       std::vector<int>& boundary_candidates,
                       StepTimings& timings) {
    const PhysicsConfig& physics_config = system.config_;
    const auto step_start = std::chrono::high_resolution_clock::now();
    timings.lambda_ms = 0.0;
    timings.correction_ms = 0.0;
    timings.apply_corrections_ms = 0.0;

    const auto gravity_start = std::chrono::high_resolution_clock::now();
    system.updateVelocityFromGravity();
    const auto gravity_end = std::chrono::high_resolution_clock::now();
    timings.gravity_ms = std::chrono::duration<double, std::milli>(gravity_end - gravity_start).count();

    const auto predict_start = std::chrono::high_resolution_clock::now();
    system.predictPositions(predicted_positions);
    const auto predict_end = std::chrono::high_resolution_clock::now();
    timings.predict_ms = std::chrono::duration<double, std::milli>(predict_end - predict_start).count();

    const auto hash_start = std::chrono::high_resolution_clock::now();
    spatial_hash.update(predicted_positions);
    const auto hash_end = std::chrono::high_resolution_clock::now();
    timings.hash_update_ms = std::chrono::duration<double, std::milli>(hash_end - hash_start).count();

    const auto neighbor_start = std::chrono::high_resolution_clock::now();
    spatial_hash.getAllNeighbors(neighbors);
    boundary_neighbors.resize(system.getNumParticles());
    for (int i = 0; i < system.getNumParticles(); ++i) {
        boundary_hash.getNeighborsForPosition(predicted_positions[i], boundary_candidates);
        boundary_neighbors[i] = boundary_candidates;
    }
    const auto neighbor_end = std::chrono::high_resolution_clock::now();
    timings.neighbor_ms = std::chrono::duration<double, std::milli>(neighbor_end - neighbor_start).count();

    total_corrections.clear();
    total_corrections.resize(system.getNumParticles(), vec2f::zero());
    lambdas.clear();
    lambdas.resize(system.getNumParticles(), 0.0f);

    for (int iter = 0; iter < solver_config.numIterations; ++iter) {
        const auto lambda_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < system.getNumParticles(); ++i) {
            lambdas[i] = sph::computeLambdaWithBoundary<2, sph::CubicSpline<2>, sph::CubicSpline<2>>(
                i, physics_config.restDensity, physics_config.mass,
                physics_config.kernelRadius, physics_config.constraintEpsilon,
                neighbors[i], boundary_neighbors[i], predicted_positions,
                boundary_positions, boundary_psi);
        }
        const auto lambda_end = std::chrono::high_resolution_clock::now();
        timings.lambda_ms += std::chrono::duration<double, std::milli>(lambda_end - lambda_start).count();

        const auto correction_start = std::chrono::high_resolution_clock::now();
        solveConstraints(predicted_positions, neighbors, boundary_neighbors,
                         boundary_positions, boundary_psi,
                         lambdas, physics_config,
                         corrections);
        const auto correction_end = std::chrono::high_resolution_clock::now();
        timings.correction_ms += std::chrono::duration<double, std::milli>(correction_end - correction_start).count();

        const auto apply_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < system.getNumParticles(); ++i) {
            total_corrections[i] += corrections[i];
            predicted_positions[i] += corrections[i];
        }
        const auto apply_end = std::chrono::high_resolution_clock::now();
        timings.apply_corrections_ms += std::chrono::duration<double, std::milli>(apply_end - apply_start).count();
    }

    const auto update_start = std::chrono::high_resolution_clock::now();
    system.updatePositions(total_corrections);
    const auto update_end = std::chrono::high_resolution_clock::now();
    timings.update_positions_ms = std::chrono::duration<double, std::milli>(update_end - update_start).count();
    const auto xsph_start = std::chrono::high_resolution_clock::now();
    sph::computeXsphViscosity<2, sph::CubicSpline<2>>(
        physics_config.viscosity,
        physics_config.mass,
        physics_config.kernelRadius,
        system.positions_,
        neighbors,
        system.velocities_);
    const auto xsph_end = std::chrono::high_resolution_clock::now();
    timings.xsph_ms = std::chrono::duration<double, std::milli>(xsph_end - xsph_start).count();
    const auto step_end = std::chrono::high_resolution_clock::now();
    timings.total_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
}

} // namespace


int main() {
    // Create complete application configuration
    AppConfig app_config = initAppConfig2D();

    std::cout << "PBF Fluid Simulation with Visualization" << std::endl;
    std::cout << "Particles: " << (kParticleCountX * kParticleCountY) << std::endl;
    std::cout << "Kernel radius: " << app_config.physics.kernelRadius << std::endl;
    std::cout << "Time step: " << app_config.physics.timeStep << std::endl;
    std::cout << "Solver iterations: " << app_config.solver.numIterations << std::endl;

    // Initialize raylib
    InitWindow(app_config.visualization.screenWidth, app_config.visualization.screenHeight, "PBF Fluid Simulation");
    SetTargetFPS(kTargetFps);

    const std::array<int, 2> particle_counts{kParticleCountX, kParticleCountY};
    const float grid_width = static_cast<float>(particle_counts[0] - 1)
                             * app_config.physics.particleSpacing;
    const vec2f origin_offset(-0.5f * grid_width, kOriginOffsetY);
    ParticleSystem<2> system(particle_counts, app_config.physics, origin_offset);

    const float domain_size = kDomainSizeScale * app_config.physics.particleSpacing;
    const float container_width = kContainerWidthScale * domain_size;
    const float container_height = kContainerHeightScale * domain_size;
    const float boundary_spacing = app_config.physics.particleSpacing;
    vec2f min_bounds(-0.5f * container_width, 0.0f);
    vec2f max_bounds(0.5f * container_width, container_height);
    SpatialHash<2> spatial_hash(min_bounds, max_bounds, app_config.physics.kernelRadius);
    std::vector<vec2f> boundary_positions;
    std::vector<float> boundary_psi;
    SpatialHash<2> boundary_hash(min_bounds, max_bounds, app_config.physics.kernelRadius);
    initBoundaryData(container_width,
                     container_height,
                     boundary_spacing,
                     app_config.physics.kernelRadius,
                     app_config.physics.restDensity,
                     boundary_positions,
                     boundary_psi,
                     boundary_hash);

    float simulationTime = 0.0f;  // Add time tracking

    // Simple simulation + display loop
    std::vector<vec2f> predicted_positions(system.getNumParticles(), vec2f::zero());
    std::vector<std::vector<int>> neighbors(system.getNumParticles());
    std::vector<std::vector<int>> boundary_neighbors_buffer(system.getNumParticles());
    std::vector<vec2f> total_corrections(system.getNumParticles(), vec2f::zero());
    std::vector<vec2f> corrections(system.getNumParticles(), vec2f::zero());
    std::vector<float> lambdas(system.getNumParticles(), 0.0f);
    std::vector<int> boundary_candidates;
    StepTimings timings;

    while (!WindowShouldClose()) {
        runSimulationStep(system, spatial_hash, boundary_hash,
                          boundary_positions, boundary_psi,
                          app_config.solver, predicted_positions,
                          neighbors, boundary_neighbors_buffer,
                          total_corrections, corrections,
                          lambdas, boundary_candidates, timings);
        simulationTime += app_config.physics.timeStep;
        renderFrame(app_config,
                    min_bounds,
                    max_bounds,
                    system.positions_,
                    app_config.physics.kernelRadius,
                    simulationTime,
                    timings);
    }

    CloseWindow();
    return 0;
}
