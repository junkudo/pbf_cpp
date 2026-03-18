#pragma once

#include <vector>
#include "vec2f.h"
#include "raylib.h"

namespace pbf::visualization {

// Configuration structure for visualization parameters
struct VisualizationConfig {
    // Screen properties
    int screenWidth = 800;
    int screenHeight = 600;

    // Coordinate transformation
    float scale = 4000.0f;
    float offsetX = 100.0f;  // X offset from left edge (moves origin right)
    float offsetY = 500.0f;  // Y offset from top edge (moves origin up, closer to bottom)

    // Particle visualization
    float particleRadius = 0.5f;
    Color particleColor = {255, 255, 255, 255}; // WHITE

    // Origin visualization
    float originDotRadius = 3.0f;
    Color originDotColor = {255, 0, 0, 255}; // RED
    Color axisColor = {128, 128, 128, 255}; // GRAY
    Color labelColor = {128, 128, 128, 255}; // GRAY
    int labelFontSize = 10;

    // Time display
    int timeFontSize = 20;
    Color timeColor = {255, 255, 255, 255}; // WHITE
};

// Coordinate transformation functions
float simToScreenX(float simX, const VisualizationConfig& config = {});
float simToScreenY(float simY, const VisualizationConfig& config = {});
vec2f simToScreen(const vec2f& simPos, const VisualizationConfig& config = {});

// Drawing functions
void drawOrigin(const VisualizationConfig& config = {});
void drawParticles(const std::vector<vec2f>& positions, float influenceRadius, const VisualizationConfig& config = {});
void drawTime(float time, const VisualizationConfig& config = {});
void drawWalls(const vec2f& min_bounds, const vec2f& max_bounds,
               const VisualizationConfig& config = {});

} // namespace pbf::visualization
