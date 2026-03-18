#pragma once

#include "pbf/visualization.h"

namespace pbf {

/**
 * Configuration for physics simulation parameters
 */
struct PhysicsConfig {
    float particleSpacing = 0.01f;
    float kernelRadius = 0.0f;  // Will be calculated as 2.5f * particleSpacing
    float restDensity = 1.0f;
    float gravity = 9.8f;
    float timeStep = 1.0f / 240.0f;
    float mass = 0.0f;  // Will be calculated as restDensity * particleSpacing * particleSpacing
    float viscosity = 0.02f;
    float jitterFactor = 0.05f;
    float constraintEpsilon = 1.0e-6f;

    /**
     * Calculate dependent values for 2D simulations.
     */
    void calculateDerivedValues2D() {
        // 2D SPH typically uses a kernel radius ~2-3x the particle spacing.
        kernelRadius = 2.5f * particleSpacing;
        mass = restDensity * particleSpacing * particleSpacing;
    }

    /**
     * Calculate dependent values for 3D simulations.
     */
    void calculateDerivedValues3D(float kernel_radius_scale = 2.0f,
                                  float mass_scale = 0.8f) {
        kernelRadius = kernel_radius_scale * particleSpacing;
        mass = mass_scale * particleSpacing * particleSpacing * particleSpacing * restDensity;
    }
};

/**
 * Configuration for PBF solver parameters
 */
struct SolverConfig {
    float constraintEpsilon = 1.0e-6f;
    int numIterations = 15;
};

/**
 * Main application configuration container
 */
struct AppConfig {
    PhysicsConfig physics;
    SolverConfig solver;
    visualization::VisualizationConfig visualization;
};

} // namespace pbf