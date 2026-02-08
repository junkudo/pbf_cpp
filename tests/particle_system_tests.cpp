#include <array>
#include <vector>

#include "gtest/gtest.h"

#include "pbf/particle_system.h"

namespace {

TEST(ParticleSystem2DTests, InitializesGridWithExpectedCounts) {
    pbf::PhysicsConfig config;
    config.particleSpacing = 0.1f;
    config.jitterFactor = 0.0f;

    const std::array<int, 2> counts{2, 3};
    const auto origin = pbf::vec2f::zero();
    pbf::ParticleSystem<2> system(counts, config, origin);

    EXPECT_EQ(system.getNumParticles(), 6);
    EXPECT_EQ(system.positions_.size(), 6u);
    EXPECT_EQ(system.velocities_.size(), 6u);
}

TEST(ParticleSystem2DTests, PredictPositionsAdvancesWithVelocity) {
    pbf::PhysicsConfig config;
    config.particleSpacing = 0.1f;
    config.jitterFactor = 0.0f;
    config.timeStep = 0.05f;

    const std::array<int, 2> counts{1, 1};
    pbf::ParticleSystem<2> system(counts, config, pbf::vec2f::zero());
    system.velocities_[0] = pbf::vec2f(1.0f, -2.0f);

    const auto predicted = system.predictPositions();

    EXPECT_FLOAT_EQ(predicted[0].x, system.positions_[0].x + config.timeStep);
    EXPECT_FLOAT_EQ(predicted[0].y, system.positions_[0].y - 2.0f * config.timeStep);
}

TEST(ParticleSystem2DTests, UpdatePositionsAppliesCorrections) {
    pbf::PhysicsConfig config;
    config.particleSpacing = 0.1f;
    config.jitterFactor = 0.0f;
    config.timeStep = 0.02f;

    const std::array<int, 2> counts{1, 1};
    pbf::ParticleSystem<2> system(counts, config, pbf::vec2f::zero());

    std::vector<pbf::vec2f> corrections{pbf::vec2f(0.1f, -0.2f)};
    system.updatePositions(corrections);

    EXPECT_FLOAT_EQ(system.positions_[0].x, 0.1f);
    EXPECT_FLOAT_EQ(system.positions_[0].y, -0.2f);
    EXPECT_FLOAT_EQ(system.velocities_[0].x, corrections[0].x / config.timeStep);
    EXPECT_FLOAT_EQ(system.velocities_[0].y, corrections[0].y / config.timeStep);
}

TEST(ParticleSystem2DTests, GravityUpdatesVelocityAlongY) {
    pbf::PhysicsConfig config;
    config.particleSpacing = 0.1f;
    config.jitterFactor = 0.0f;
    config.timeStep = 0.1f;
    config.gravity = 9.8f;

    const std::array<int, 2> counts{1, 1};
    pbf::ParticleSystem<2> system(counts, config, pbf::vec2f::zero());
    system.updateVelocityFromGravity();

    EXPECT_FLOAT_EQ(system.velocities_[0].y, -config.gravity * config.timeStep);
}

TEST(ParticleSystem3DTests, InitializesGridWithExpectedCounts) {
    pbf::PhysicsConfig config;
    config.particleSpacing = 0.2f;
    config.jitterFactor = 0.0f;

    const std::array<int, 3> counts{2, 1, 2};
    const auto origin = pbf::vec3f::zero();
    pbf::ParticleSystem<3> system(counts, config, origin);

    EXPECT_EQ(system.getNumParticles(), 4);
    EXPECT_EQ(system.positions_.size(), 4u);
    EXPECT_EQ(system.velocities_.size(), 4u);
}

TEST(ParticleSystem3DTests, PredictPositionsAdvancesWithVelocity) {
    pbf::PhysicsConfig config;
    config.particleSpacing = 0.2f;
    config.jitterFactor = 0.0f;
    config.timeStep = 0.03f;

    const std::array<int, 3> counts{1, 1, 1};
    pbf::ParticleSystem<3> system(counts, config, pbf::vec3f::zero());
    system.velocities_[0] = pbf::vec3f(1.5f, -2.5f, 0.5f);

    const auto predicted = system.predictPositions();

    EXPECT_FLOAT_EQ(predicted[0].x, system.positions_[0].x + 1.5f * config.timeStep);
    EXPECT_FLOAT_EQ(predicted[0].y, system.positions_[0].y - 2.5f * config.timeStep);
    EXPECT_FLOAT_EQ(predicted[0].z, system.positions_[0].z + 0.5f * config.timeStep);
}

TEST(ParticleSystem3DTests, UpdatePositionsAppliesCorrections) {
    pbf::PhysicsConfig config;
    config.particleSpacing = 0.2f;
    config.jitterFactor = 0.0f;
    config.timeStep = 0.04f;

    const std::array<int, 3> counts{1, 1, 1};
    pbf::ParticleSystem<3> system(counts, config, pbf::vec3f::zero());

    std::vector<pbf::vec3f> corrections{pbf::vec3f(0.2f, -0.1f, 0.3f)};
    system.updatePositions(corrections);

    EXPECT_FLOAT_EQ(system.positions_[0].x, 0.2f);
    EXPECT_FLOAT_EQ(system.positions_[0].y, -0.1f);
    EXPECT_FLOAT_EQ(system.positions_[0].z, 0.3f);
    EXPECT_FLOAT_EQ(system.velocities_[0].x, corrections[0].x / config.timeStep);
    EXPECT_FLOAT_EQ(system.velocities_[0].y, corrections[0].y / config.timeStep);
    EXPECT_FLOAT_EQ(system.velocities_[0].z, corrections[0].z / config.timeStep);
}

TEST(ParticleSystem3DTests, GravityUpdatesVelocityAlongY) {
    pbf::PhysicsConfig config;
    config.particleSpacing = 0.2f;
    config.jitterFactor = 0.0f;
    config.timeStep = 0.05f;
    config.gravity = 4.5f;

    const std::array<int, 3> counts{1, 1, 1};
    pbf::ParticleSystem<3> system(counts, config, pbf::vec3f::zero());
    system.updateVelocityFromGravity();

    EXPECT_FLOAT_EQ(system.velocities_[0].y, -config.gravity * config.timeStep);
}

}