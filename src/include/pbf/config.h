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
     * Calculate dependent values based on the current configuration
     */
    void calculateDerivedValues() {
        kernelRadius = 2.5f * particleSpacing;
        mass = restDensity * particleSpacing * particleSpacing;
    }
};

/**
 * Configuration for PBF solver parameters
 */
struct SolverConfig {
    float constraintEpsilon = 1.0e-6f;
    float positionCorrectionFactor = 0.1f;
    int numIterations = 50;
};

/**
 * Main application configuration container
 */
struct AppConfig {
    PhysicsConfig physics;
    SolverConfig solver;
    visualization::VisualizationConfig visualization;

    /**
     * Initialize with default values and calculate derived physics values
     */
    AppConfig() {
        physics.calculateDerivedValues();
    }
};

} // namespace pbf