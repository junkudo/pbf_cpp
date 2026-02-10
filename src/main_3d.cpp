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
#include "pbf/vec3f.h"
#include "pbf/config.h"
#include "pbf/lighting_shaders.h"
#include "raylib.h"
#include "raymath.h"

#define RLIGHTS_IMPLEMENTATION
#include "rlights.h"

using namespace pbf;

namespace {

constexpr int kParticlesPerAxis = 10;
constexpr float kVisualizationScale = 10.0f;
constexpr float kCameraDistanceScale = 3.0f;
constexpr float kDrawRadiusScale = 0.01f;
constexpr float kAmbientLight = 0.05f;
constexpr float kShaderLightHeight = 0.4f;
constexpr float kShaderLightDistance = 0.4f;

Vector3 toRaylib(const vec3f& position) {
    return Vector3{
        position.x * kVisualizationScale,
        position.y * kVisualizationScale,
        position.z * kVisualizationScale
    };
}

Color particleColor() {
    return Color{200, 200, 220, 255};
}

float calculateLambda(int i, const std::vector<int>& neighbors,
                      const std::vector<vec3f>& positions, const PhysicsConfig& config) {
    return sph::computeLambda<3, sph::Poly6<3>, sph::Spikey<3>>(i, config.restDensity, config.mass,
                                                                config.kernelRadius, config.constraintEpsilon,
                                                                neighbors, positions);
}

std::vector<vec3f> solveConstraints(const std::vector<vec3f>& predicted_positions,
                                   const std::vector<std::vector<int>>& neighbors,
                                   const std::vector<float>& lambdas, const PhysicsConfig& config) {
    std::vector<vec3f> corrections(neighbors.size(), vec3f::zero());

    // Calculate position corrections
    for (int i = 0; i < neighbors.size(); ++i) {
        sph::calculatePositionCorrection<3, sph::Spikey<3>>(i, config.restDensity, config.mass, config.kernelRadius,
                                                             neighbors[i], lambdas, predicted_positions,
                                                             corrections[i]);
    }

    return corrections;
}

void runSimulationStep(ParticleSystem<3>& system, SpatialHash<3>& spatial_hash, const SolverConfig& solver_config) {
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
    std::vector<vec3f> total_corrections(system.getNumParticles(), vec3f::zero());
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

} // namespace

int main() {
    // Create complete application configuration
    AppConfig app_config;

    std::cout << "PBF Fluid Simulation 3D with Visualization" << std::endl;
    std::cout << "Particles: " << (kParticlesPerAxis * kParticlesPerAxis * kParticlesPerAxis) << std::endl;
    std::cout << "Kernel radius: " << app_config.physics.kernelRadius << std::endl;
    std::cout << "Time step: " << app_config.physics.timeStep << std::endl;
    std::cout << "Solver iterations: " << app_config.solver.numIterations << std::endl;

    // Initialize raylib
    InitWindow(app_config.visualization.screenWidth, app_config.visualization.screenHeight, "PBF Fluid Simulation 3D");

    const std::array<int, 3> particle_counts{kParticlesPerAxis, kParticlesPerAxis, kParticlesPerAxis};
    const vec3f origin_offset(0.0f, 0.05f, 0.0f);
    ParticleSystem<3> system(particle_counts, app_config.physics, origin_offset);

    // Build spatial hash
    float domain_size = static_cast<float>(kParticlesPerAxis) * app_config.physics.particleSpacing;
    vec3f min_bounds(-domain_size, -domain_size, -domain_size);
    vec3f max_bounds(domain_size, domain_size, domain_size);
    SpatialHash<3> spatial_hash(min_bounds, max_bounds, app_config.physics.kernelRadius);

    // Setup 3D camera
    const float camera_distance = domain_size * kVisualizationScale * kCameraDistanceScale;
    Camera3D camera{};
    camera.position = { camera_distance, camera_distance, camera_distance };
    camera.target = { 0.0f, 0.0f, 0.0f };
    camera.up = { 0.0f, 1.0f, 0.0f };
    camera.fovy = 60.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    Shader shader = LoadShaderFromMemory(pbf::visualization::kLightingVertexShader,
                                         pbf::visualization::kLightingFragmentShader);
    if (shader.id == 0) {
        TraceLog(LOG_ERROR, "Lighting shader failed to compile");
    }
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
    const int step_count = 10;
    const bool freeze_simulation = true;
    const float box_extent = domain_size * kVisualizationScale;

    Model sphere_model = LoadModelFromMesh(GenMeshSphere(draw_radius, 16, 16));
    sphere_model.materials[0].shader = shader;
    sphere_model.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = particleColor();

    SetTargetFPS(60);

    for (int step = 0; step < step_count; ++step) {
        runSimulationStep(system, spatial_hash, app_config.solver);
        simulation_time += app_config.physics.timeStep;
    }

    while (!WindowShouldClose()) {
        if (!freeze_simulation) {
            runSimulationStep(system, spatial_hash, app_config.solver);
            simulation_time += app_config.physics.timeStep;
        }

        BeginDrawing();
        ClearBackground(BLACK);

        BeginMode3D(camera);

        DrawGrid(10, box_extent / 5.0f);
        DrawCubeWires({0.0f, 0.0f, 0.0f}, box_extent * 2.0f, box_extent * 2.0f, box_extent * 2.0f, GRAY);
        DrawSphere({0.0f, 0.0f, 0.0f}, draw_radius * 1.5f, RED);

        float cameraPos[3] = { camera.position.x, camera.position.y, camera.position.z };
        SetShaderValue(shader, shader.locs[SHADER_LOC_VECTOR_VIEW], cameraPos, SHADER_UNIFORM_VEC3);
        for (int i = 0; i < MAX_LIGHTS; ++i) {
            UpdateLightValues(shader, lights[i]);
        }

        BeginShaderMode(shader);
        DrawPlane(Vector3Zero(), (Vector2){ box_extent * 1.2f, box_extent * 1.2f }, WHITE);
        DrawCube(Vector3Zero(), box_extent * 0.5f, box_extent * 0.8f, box_extent * 0.5f, WHITE);
        for (const auto& position : system.positions_) {
            const auto center = toRaylib(position);
            DrawModel(sphere_model, center, 1.0f, WHITE);
        }
        EndShaderMode();

        EndMode3D();

        DrawText(TextFormat("Time: %.3f s", simulation_time), 10, 10, 20, RAYWHITE);
        DrawText("PBF 3D (frozen)", 10, 40, 20, RAYWHITE);

        EndDrawing();
    }

    UnloadModel(sphere_model);
    UnloadShader(shader);
    CloseWindow();
    return 0;
}
