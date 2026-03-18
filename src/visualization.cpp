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

        // White center dot removed to emphasize influence radius only.
    }
}

void drawTime(float time, const VisualizationConfig& config) {
    std::string timeText = "Time: " + std::to_string(time) + " s";
    const std::string fpsText = "FPS: " + std::to_string(GetFPS());
    const int time_width = MeasureText(timeText.c_str(), config.timeFontSize);
    const int fps_width = MeasureText(fpsText.c_str(), config.timeFontSize);
    const int x_time = config.screenWidth - time_width - 10;
    const int x_fps = config.screenWidth - fps_width - 10;
    const int y = 10;
    const int line_spacing = config.timeFontSize + 4;
    DrawText(timeText.c_str(), x_time, y, config.timeFontSize, config.timeColor);
    DrawText(fpsText.c_str(), x_fps, y + line_spacing, config.timeFontSize, config.timeColor);
}

void drawWalls(const vec2f& min_bounds, const vec2f& max_bounds, const VisualizationConfig& config) {
    const vec2f bottom_left = simToScreen(vec2f(min_bounds.x, min_bounds.y), config);
    const vec2f bottom_right = simToScreen(vec2f(max_bounds.x, min_bounds.y), config);
    const vec2f top_left = simToScreen(vec2f(min_bounds.x, max_bounds.y), config);
    const vec2f top_right = simToScreen(vec2f(max_bounds.x, max_bounds.y), config);
    DrawLine(bottom_left.x, bottom_left.y, bottom_right.x, bottom_right.y, BLUE);
    DrawLine(bottom_left.x, bottom_left.y, top_left.x, top_left.y, BLUE);
    DrawLine(bottom_right.x, bottom_right.y, top_right.x, top_right.y, BLUE);
}

} // namespace pbf::visualization
