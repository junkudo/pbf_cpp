#include <gtest/gtest.h>
#include "pbf/vec2f.h"

// Test the coordinate transformation math with custom VisualizationConfig
TEST(VisualizationTest, CoordinateTransformation) {
    // Define test configuration with offset
    struct VisualizationConfig {
        int screenWidth = 800;
        int screenHeight = 600;
        float scale = 4000.0f;
        float offsetX = 100.0f;   // Move origin 100 pixels from left edge
        float offsetY = 500.0f;   // Move origin 500 pixels from top (closer to bottom)
    };

    // Coordinate transformation functions (implementing the math with offset)
    auto simToScreenX = [](float simX, const VisualizationConfig& config) {
        return config.offsetX + (simX * config.scale);
    };

    auto simToScreenY = [](float simY, const VisualizationConfig& config) {
        return config.offsetY - (simY * config.scale);
    };

    auto simToScreen = [&](const pbf::vec2f& simPos, const VisualizationConfig& config) {
        return pbf::vec2f(simToScreenX(simPos.x, config), simToScreenY(simPos.y, config));
    };

    // Create custom visualization configuration
    VisualizationConfig config;

    // Named variables showing the math
    float TEST_SIM_X = 0.01f;
    float TEST_SIM_Y = 0.01f;

    // Expected values calculated with the coordinate transformation math:
    // simToScreenX(x) = offsetX + (x * scale)
    // simToScreenY(y) = offsetY - (y * scale)
    //
    // For test input (0.01, 0.01):
    // X: 100 + (0.01 * 4000) = 100 + 40 = 140
    // Y: 500 - (0.01 * 4000) = 500 - 40 = 460
    float EXPECTED_SCREEN_X = config.offsetX + (TEST_SIM_X * config.scale); // 140
    float EXPECTED_SCREEN_Y = config.offsetY - (TEST_SIM_Y * config.scale); // 460

    // Test center point transformation (origin at offset)
    EXPECT_FLOAT_EQ(simToScreenX(0.0f, config), config.offsetX);  // 100
    EXPECT_FLOAT_EQ(simToScreenY(0.0f, config), config.offsetY);  // 500

    // Test simple coordinate transformation with math shown
    EXPECT_FLOAT_EQ(simToScreenX(TEST_SIM_X, config), EXPECTED_SCREEN_X);
    EXPECT_FLOAT_EQ(simToScreenY(TEST_SIM_Y, config), EXPECTED_SCREEN_Y);

    // Test vec2f transformation consistency
    pbf::vec2f simPos(TEST_SIM_X, TEST_SIM_Y);
    pbf::vec2f screenPos = simToScreen(simPos, config);
    EXPECT_FLOAT_EQ(screenPos.x, EXPECTED_SCREEN_X);
    EXPECT_FLOAT_EQ(screenPos.y, EXPECTED_SCREEN_Y);
}
