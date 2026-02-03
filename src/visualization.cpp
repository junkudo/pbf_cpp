#include "pbf/visualization.h"
#include "raylib.h"
#include <string>

namespace pbf::visualization {

// Coordinate transformation functions
float simToScreenX(float simX, const VisualizationConfig& config) {
    return config.offsetX + (simX * config.scale);
}

float simToScreenY(float simY, const VisualizationConfig& config) {
    return config.offsetY - (simY * config.scale);
}

vec2f simToScreen(const vec2f& simPos, const VisualizationConfig& config) {
    return vec2f(simToScreenX(simPos.x, config), simToScreenY(simPos.y, config));
}

// Drawing functions
void drawOrigin(const VisualizationConfig& config) {
    // Draw origin as a small red dot at the configured offset position
    DrawCircle(config.offsetX, config.offsetY, config.originDotRadius, config.originDotColor);

    // Draw coordinate axes from the origin
    DrawLine(config.offsetX, 0, config.offsetX, config.screenHeight, config.axisColor);  // Y-axis
    DrawLine(0, config.offsetY, config.screenWidth, config.offsetY, config.axisColor);   // X-axis

    // Label the origin
    DrawText("Origin (0,0)", config.offsetX + 10, config.offsetY + 10, config.labelFontSize, config.labelColor);
}

void drawParticles(const std::vector<vec2f>& positions, float influenceRadius, const VisualizationConfig& config) {
    for (int i = 0; i < positions.size(); ++i) {
        const auto& pos = positions[i];
        auto screenPos = simToScreen(pos, config);

        // Draw influence circle (blue, semi-transparent)
        float screenInfluenceRadius = influenceRadius * config.scale;
        DrawCircle(screenPos.x, screenPos.y, screenInfluenceRadius, Fade(BLUE, 0.3f));

        // Draw particle as a white dot
        DrawCircle(screenPos.x, screenPos.y, config.particleRadius, config.particleColor);
    }
}

void drawTime(float time, const VisualizationConfig& config) {
    std::string timeText = "Time: " + std::to_string(time) + " s";
    DrawText(timeText.c_str(), 10, 10, config.timeFontSize, config.timeColor);
}

} // namespace pbf::visualization
