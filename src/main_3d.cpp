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
#include "pbf/vec3f.h"
#include "pbf/config.h"
#include "pbf/lighting_shaders.h"
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#define RLIGHTS_IMPLEMENTATION
#include "rlights.h"

using namespace pbf;

namespace {

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
    double render_walls_ms = 0.0;
    double render_particles_ms = 0.0;
    double render_overlay_ms = 0.0;
    double render_end_ms = 0.0;
};

constexpr bool kUseLegacyInit = true;
constexpr int kParticlesWidth = kUseLegacyInit ? 15 : 8;
constexpr int kParticlesHeight = kUseLegacyInit ? 20 : 10;
constexpr int kParticlesDepth = kUseLegacyInit ? 15 : 8;
constexpr float kDenseSpacingScale = 0.7f;
constexpr float kLegacySpacingScale = 1.0f;
constexpr float kParticleSpacing = 0.05f;
constexpr float kVisualizationScale = 10.0f;
constexpr float kCameraDistanceScale = 1.0f;
constexpr float kDrawRadiusScale = 0.1f;
constexpr float kAmbientLight = 0.05f;
constexpr float kShaderLightHeight = 0.4f;
constexpr float kShaderLightDistance = 0.4f;
constexpr bool kDisableBoundaryParticles = false;

enum class InitialConditionMode {
    SubRestDensity,
    SuperRestDensity
};

constexpr InitialConditionMode kInitialConditionMode = InitialConditionMode::SuperRestDensity;

void addWall(const vec3f& min_corner, const vec3f& max_corner,
             float particle_distance,
             std::vector<vec3f>& boundary_positions) {
    const vec3f diff = max_corner - min_corner;
    const int steps_x = static_cast<int>(diff.x / particle_distance) + 1;
    const int steps_y = static_cast<int>(diff.y / particle_distance) + 1;
    const int steps_z = static_cast<int>(diff.z / particle_distance) + 1;

    const int start_index = static_cast<int>(boundary_positions.size());
    boundary_positions.resize(start_index + steps_x * steps_y * steps_z, vec3f::zero());

    for (int i = 0; i < steps_x; ++i) {
        for (int j = 0; j < steps_y; ++j) {
            for (int k = 0; k < steps_z; ++k) {
                const vec3f curr_pos = min_corner + vec3f(static_cast<float>(i),
                                                         static_cast<float>(j),
                                                         static_cast<float>(k))
                                                     * particle_distance;
                boundary_positions[start_index + i * steps_y * steps_z + j * steps_z + k] = curr_pos;
            }
        }
    }
}

std::vector<vec3f> initContainerBoundary(float container_width,
                                         float container_depth,
                                         float container_height,
                                         float particle_distance) {
    const float x1 = -0.5f * container_width;
    const float x2 = 0.5f * container_width;
    const float y1 = 0.0f;
    const float y2 = container_height;
    const float z1 = -0.5f * container_depth;
    const float z2 = 0.5f * container_depth;

    std::vector<vec3f> boundary_positions;
    addWall(vec3f(x1, y1, z1), vec3f(x2, y1, z2), particle_distance, boundary_positions);
    addWall(vec3f(x1, y2, z1), vec3f(x2, y2, z2), particle_distance, boundary_positions);
    addWall(vec3f(x1, y1, z1), vec3f(x1, y2, z2), particle_distance, boundary_positions);
    addWall(vec3f(x2, y1, z1), vec3f(x2, y2, z2), particle_distance, boundary_positions);
    addWall(vec3f(x1, y1, z1), vec3f(x2, y2, z1), particle_distance, boundary_positions);
    addWall(vec3f(x1, y1, z2), vec3f(x2, y2, z2), particle_distance, boundary_positions);
    return boundary_positions;
}


std::vector<vec3f> createFluidDemoGridPositions(const std::array<int, 3>& counts,
                                                float particle_spacing,
                                                const vec3f& origin_offset) {
    const int width = counts[0];
    const int height = counts[1];
    const int depth = counts[2];
    std::vector<vec3f> positions;
    positions.resize(width * height * depth, vec3f::zero());

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < depth; ++k) {
                const int index = i * height * depth + j * depth + k;
                positions[index] = vec3f(static_cast<float>(i),
                                         static_cast<float>(j),
                                         static_cast<float>(k)) * particle_spacing
                                   + origin_offset;
            }
        }
    }

    return positions;
}


Color particleColor() {
    return Color{70, 130, 255, 255};
}


void solveConstraints(const std::vector<vec3f>& predicted_positions,
                      const std::vector<std::vector<int>>& neighbors,
                      const std::vector<std::vector<int>>& boundary_neighbors,
                      const std::vector<vec3f>& boundary_positions,
                      const std::vector<float>& boundary_psi,
                      const std::vector<float>& lambdas,
                      const PhysicsConfig& config,
                      std::vector<vec3f>& corrections) {
    corrections.clear();
    corrections.resize(neighbors.size(), vec3f::zero());

    // Calculate position corrections
    for (int i = 0; i < static_cast<int>(neighbors.size()); ++i) {
        sph::calculatePositionCorrection<3, sph::CubicSpline<3>>(
            i, config.restDensity, config.mass, config.kernelRadius,
            neighbors[i], lambdas, predicted_positions, corrections[i]);
        sph::computePositionCorrectionBoundary<3, sph::CubicSpline<3>>(
            i, config.restDensity, config.kernelRadius,
            boundary_neighbors[i], predicted_positions, boundary_positions,
            boundary_psi, lambdas[i], corrections[i]);
    }
}

void runSimulationStep(int step_index, float time,
                       ParticleSystem<3>& system, SpatialHash<3>& spatial_hash,
                       const SpatialHash<3>& boundary_hash,
                       const std::vector<vec3f>& boundary_positions,
                       const std::vector<float>& boundary_psi,
                       const SolverConfig& solver_config,
                       std::vector<vec3f>& predicted_positions,
                       std::vector<std::vector<int>>& neighbors,
                       std::vector<std::vector<int>>& boundary_neighbors,
                       std::vector<vec3f>& total_corrections,
                       std::vector<vec3f>& corrections,
                       std::vector<float>& lambdas,
                       std::vector<int>& boundary_candidates,
                       StepTimings& timings) {
    const PhysicsConfig& physics_config = system.config_;  // Get physics config from system
    const auto step_start = std::chrono::high_resolution_clock::now();
    timings.lambda_ms = 0.0;
    timings.correction_ms = 0.0;
    timings.apply_corrections_ms = 0.0;

    // 1. Update velocities from gravity
    const auto gravity_start = std::chrono::high_resolution_clock::now();
    system.updateVelocityFromGravity();
    const auto gravity_end = std::chrono::high_resolution_clock::now();
    timings.gravity_ms = std::chrono::duration<double, std::milli>(gravity_end - gravity_start).count();

    // 2. Predict positions
    const auto predict_start = std::chrono::high_resolution_clock::now();
    system.predictPositions(predicted_positions);
    const auto predict_end = std::chrono::high_resolution_clock::now();
    timings.predict_ms = std::chrono::duration<double, std::milli>(predict_end - predict_start).count();

    // 3. Build spatial hash with predicted positions
    const auto hash_start = std::chrono::high_resolution_clock::now();
    spatial_hash.update(predicted_positions);
    const auto hash_end = std::chrono::high_resolution_clock::now();
    timings.hash_update_ms = std::chrono::duration<double, std::milli>(hash_end - hash_start).count();

    // 4. Find neighbors for each particle
    const auto neighbor_start = std::chrono::high_resolution_clock::now();
    spatial_hash.getAllNeighbors(neighbors);
    boundary_neighbors.resize(system.getNumParticles());
    if (!kDisableBoundaryParticles) {
        for (int i = 0; i < system.getNumParticles(); ++i) {
            boundary_hash.getNeighborsForPosition(predicted_positions[i], boundary_candidates);
            boundary_neighbors[i] = boundary_candidates;
        }
    }
    const auto neighbor_end = std::chrono::high_resolution_clock::now();
    timings.neighbor_ms = std::chrono::duration<double, std::milli>(neighbor_end - neighbor_start).count();

    // 5. Solve constraints (iterations from solver config)
    total_corrections.clear();
    total_corrections.resize(system.getNumParticles(), vec3f::zero());
    lambdas.clear();
    lambdas.resize(system.getNumParticles(), 0.0f);

    for (int iter = 0; iter < solver_config.numIterations; ++iter) {
        // Calculate lambdas
        const auto lambda_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < system.getNumParticles(); ++i) {
            lambdas[i] = sph::computeLambdaWithBoundary<3, sph::CubicSpline<3>, sph::CubicSpline<3>>(
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

        // Accumulate and apply corrections for this iteration.
        const auto apply_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < system.getNumParticles(); ++i) {
            total_corrections[i] += corrections[i];
            predicted_positions[i] += corrections[i];
        }
        const auto apply_end = std::chrono::high_resolution_clock::now();
        timings.apply_corrections_ms += std::chrono::duration<double, std::milli>(apply_end - apply_start).count();
    }

    // 6. Update velocities from position corrections
    // 7. Update positions to final positions
    const auto update_start = std::chrono::high_resolution_clock::now();
    system.updatePositions(total_corrections);
    const auto update_end = std::chrono::high_resolution_clock::now();
    timings.update_positions_ms = std::chrono::duration<double, std::milli>(update_end - update_start).count();
    // 8. Apply XSPH viscosity to smooth velocities.
    const auto xsph_start = std::chrono::high_resolution_clock::now();
    sph::computeXsphViscosity<3, sph::CubicSpline<3>>(
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
    AppConfig app_config;

    const float spacing_scale = kUseLegacyInit
                                    ? kLegacySpacingScale
                                    : (kInitialConditionMode == InitialConditionMode::SuperRestDensity)
                                          ? kDenseSpacingScale
                                          : 1.0f;
    app_config.physics.particleSpacing = kParticleSpacing * spacing_scale;
    app_config.physics.restDensity = 1000.0f;
    app_config.physics.calculateDerivedValues3D(2.0f, 0.8f);
    app_config.physics.jitterFactor = 0.0f;
    app_config.solver.numIterations = 5;


    // Initialize raylib
    InitWindow(app_config.visualization.screenWidth, app_config.visualization.screenHeight, "PBF Fluid Simulation 3D");
    rlEnableBackfaceCulling();

    const std::array<int, 3> particle_counts{kParticlesWidth, kParticlesHeight, kParticlesDepth};
    const float container_width = (static_cast<float>(kParticlesWidth) + 1.0f)
                                  * kParticleSpacing * 5.0f;
    const float container_depth = (static_cast<float>(kParticlesDepth) + 1.0f)
                                  * kParticleSpacing;
    const float container_height = 4.0f;
    const float start_x = -0.5f * container_width + kParticleSpacing;
    const float start_y = kParticleSpacing;
    const float start_z = -0.5f * container_depth + kParticleSpacing;
    const vec3f origin_offset(start_x, start_y, start_z);
    auto initial_positions = createFluidDemoGridPositions(particle_counts,
                                                          app_config.physics.particleSpacing,
                                                          origin_offset);
    ParticleSystem<3> system(initial_positions, app_config.physics);

    // Build spatial hash
    const float domain_size = std::max(container_width, container_depth);
    vec3f min_bounds(-0.5f * container_width, 0.0f, -0.5f * container_depth);
    vec3f max_bounds(0.5f * container_width, container_height, 0.5f * container_depth);
    SpatialHash<3> spatial_hash(min_bounds, max_bounds, app_config.physics.kernelRadius);
    const float boundary_spacing = kParticleSpacing;
    std::vector<vec3f> boundary_positions;
    std::vector<float> boundary_psi;
    SpatialHash<3> boundary_hash(min_bounds, max_bounds, app_config.physics.kernelRadius);
    if (!kDisableBoundaryParticles) {
        std::vector<int> boundary_neighbors;
        boundary_positions = initContainerBoundary(container_width, container_depth, container_height,
                                                   boundary_spacing);
        boundary_psi.resize(boundary_positions.size(), 0.0f);
        boundary_hash.update(boundary_positions);
        for (size_t i = 0; i < boundary_psi.size(); ++i) {
            const vec3f& pi = boundary_positions[i];
            boundary_hash.getNeighborsForPosition(pi, boundary_neighbors);
            boundary_psi[i] = sph::computeBoundaryPsi<3, sph::CubicSpline<3>>(
                static_cast<int>(i),
                app_config.physics.kernelRadius,
                app_config.physics.restDensity,
                1.0f, // Boundary pressure strength.
                boundary_neighbors,
                boundary_positions);
        }
    }

    // Setup 3D camera
    const float camera_distance = domain_size * kVisualizationScale * kCameraDistanceScale;
    Camera3D camera{};
    camera.position = { camera_distance, camera_distance, camera_distance };
    camera.target = { 0.0f, container_height * 0.5f, 0.0f };
    camera.up = { 0.0f, 1.0f, 0.0f };
    camera.fovy = 60.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    Shader shader = LoadShaderFromMemory(pbf::visualization::kLightingVertexShader,
                                         pbf::visualization::kLightingFragmentShader);
    if (shader.id == 0) {
        TraceLog(LOG_ERROR, "Lighting shader failed to compile");
    }
    shader.locs[SHADER_LOC_VERTEX_INSTANCE_TX] = GetShaderLocationAttrib(shader, "instanceTransform");
    shader.locs[SHADER_LOC_VECTOR_VIEW] = GetShaderLocation(shader, "viewPos");
    int ambientLoc = GetShaderLocation(shader, "ambient");
    float ambient[4] = { kAmbientLight, kAmbientLight, kAmbientLight, 1.0f };
    SetShaderValue(shader, ambientLoc, ambient, SHADER_UNIFORM_VEC4);

    Light lights[MAX_LIGHTS] = { 0 };
    lights[0] = CreateLight(LIGHT_DIRECTIONAL,
                            (Vector3){ camera_distance * 0.5f, camera_distance * 0.4f, camera_distance * 0.2f },
                            (Vector3){ -1.0f, -0.4f, -0.2f }, WHITE, shader);

    float simulation_time = 0.0f;
    const float draw_radius = app_config.visualization.particleRadius * kVisualizationScale * kDrawRadiusScale;
    const Vector3 wall_center{0.0f,
                              container_height * 0.5f * kVisualizationScale,
                              0.0f};
    const Vector3 wall_size{container_width * kVisualizationScale,
                            container_height * kVisualizationScale,
                            container_depth * kVisualizationScale};
    const Color wall_color{120, 160, 200, 80};

    Mesh sphere_mesh = GenMeshSphere(draw_radius, 4, 4);
    UploadMesh(&sphere_mesh, false);
    Material sphere_material = LoadMaterialDefault();
    sphere_material.shader = shader;
    sphere_material.maps[MATERIAL_MAP_DIFFUSE].color = particleColor();

    SetTargetFPS(60);

    std::vector<vec3f> predicted_positions(system.getNumParticles(), vec3f::zero());
    std::vector<std::vector<int>> neighbors(system.getNumParticles());
    std::vector<std::vector<int>> boundary_neighbors(system.getNumParticles());
    std::vector<vec3f> total_corrections(system.getNumParticles(), vec3f::zero());
    std::vector<vec3f> corrections(system.getNumParticles(), vec3f::zero());
    std::vector<float> lambdas(system.getNumParticles(), 0.0f);
    std::vector<int> boundary_candidates;
    std::vector<Matrix> particle_transforms(system.getNumParticles());
    const int sim_steps_per_render = 4;

    int step = 0;
    StepTimings timings;
    while (!WindowShouldClose()) {
        for (int sim_step = 0; sim_step < sim_steps_per_render; ++sim_step) {
            runSimulationStep(step, simulation_time, system, spatial_hash,
                              boundary_hash, boundary_positions, boundary_psi,
                              app_config.solver, predicted_positions,
                              neighbors, boundary_neighbors,
                              total_corrections, corrections,
                              lambdas, boundary_candidates, timings);
            simulation_time += app_config.physics.timeStep;
            ++step;
        }

        timings.render_begin_ms = 0.0;
        timings.render_walls_ms = 0.0;
        timings.render_particles_ms = 0.0;
        timings.render_overlay_ms = 0.0;
        timings.render_end_ms = 0.0;

        const auto render_start = std::chrono::high_resolution_clock::now();
        const auto render_begin_start = std::chrono::high_resolution_clock::now();
        BeginDrawing();
        ClearBackground(RAYWHITE);
        const auto render_begin_end = std::chrono::high_resolution_clock::now();
        timings.render_begin_ms = std::chrono::duration<double, std::milli>(render_begin_end - render_begin_start).count();

        BeginMode3D(camera);
        const auto render_walls_start = std::chrono::high_resolution_clock::now();
        DrawCube(wall_center, wall_size.x, wall_size.y, wall_size.z, wall_color);
        DrawCubeWires(wall_center, wall_size.x, wall_size.y, wall_size.z, BLUE);
        const auto render_walls_end = std::chrono::high_resolution_clock::now();
        timings.render_walls_ms = std::chrono::duration<double, std::milli>(render_walls_end - render_walls_start).count();
        const auto render_particles_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < system.positions_.size(); ++i) {
            const auto& position = system.positions_[i];
            const Vector3 sphere_pos{position.x * kVisualizationScale,
                                      position.y * kVisualizationScale,
                                      position.z * kVisualizationScale};
            particle_transforms[i] = MatrixTranslate(sphere_pos.x, sphere_pos.y, sphere_pos.z);
        }
        DrawMeshInstanced(sphere_mesh, sphere_material, particle_transforms.data(),
                          static_cast<int>(particle_transforms.size()));
        const auto render_particles_end = std::chrono::high_resolution_clock::now();
        timings.render_particles_ms = std::chrono::duration<double, std::milli>(render_particles_end - render_particles_start).count();
        EndMode3D();

        const auto render_overlay_start = std::chrono::high_resolution_clock::now();
        const int overlay_x = 10;
        const int overlay_y = 10;
        const int overlay_size = 18;
        const int overlay_line = 20;
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
        DrawText(TextFormat("  Walls: %.2f ms", timings.render_walls_ms),
                 overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
        DrawText(TextFormat("  Particles: %.2f ms", timings.render_particles_ms),
                 overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
        DrawText(TextFormat("  Overlay text: %.2f ms", timings.render_overlay_ms),
                 overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
        DrawText(TextFormat("  EndDrawing: %.2f ms", timings.render_end_ms),
                 overlay_x, overlay_y + overlay_line * line_index++, overlay_size, DARKGRAY);
        const auto render_overlay_end = std::chrono::high_resolution_clock::now();
        timings.render_overlay_ms = std::chrono::duration<double, std::milli>(render_overlay_end - render_overlay_start).count();
        const auto render_end_start = std::chrono::high_resolution_clock::now();
        EndDrawing();
        const auto render_end = std::chrono::high_resolution_clock::now();
        timings.render_end_ms = std::chrono::duration<double, std::milli>(render_end - render_end_start).count();
        timings.render_ms = std::chrono::duration<double, std::milli>(render_end - render_start).count();

    }

    CloseWindow();
    return 0;

}
