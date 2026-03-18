#include <gtest/gtest.h>
#include <random>
#include <chrono>
#include "pbf/spatial_hash.h"
#include "fixtures/test_helpers.h"

namespace {

// Helper function for brute force neighbor search (2D)
std::vector<int> getNeighborsBruteForce2D(int self_index,
                                          const std::vector<pbf::vec2f>& positions,
                                          float h) {
    std::vector<int> neighbors;
    const pbf::vec2f& position_i = positions.at(self_index);

    for (size_t j = 0; j < positions.size(); ++j) {
        if (static_cast<int>(j) == self_index) continue;

        const pbf::vec2f& position_j = positions.at(j);
        pbf::vec2f diff = position_i - position_j;
        float distance = diff.length();

        if (distance <= h) {
            neighbors.push_back(static_cast<int>(j));
        }
    }

    return neighbors;
}

// Helper function for brute force neighbor search (3D)
std::vector<int> getNeighborsBruteForce3D(int self_index,
                                          const std::vector<pbf::vec3f>& positions,
                                          float h) {
    std::vector<int> neighbors;
    const pbf::vec3f& position_i = positions.at(self_index);

    for (size_t j = 0; j < positions.size(); ++j) {
        if (static_cast<int>(j) == self_index) continue;

        const pbf::vec3f& position_j = positions.at(j);
        pbf::vec3f diff = position_i - position_j;
        float distance = diff.length();

        if (distance <= h) {
            neighbors.push_back(static_cast<int>(j));
        }
    }

    return neighbors;
}

// Test fixture for spatial hash tests
struct SpatialHashTest : ::testing::Test {
    std::mt19937 rng{42};  // Fixed seed for reproducible tests
    float h = 1.0f;
    float domain_size = 10.0f;

    // Create random particle positions
    std::vector<pbf::vec2f> createRandomParticles(int count) {
        std::uniform_real_distribution<float> dist(0.0f, domain_size);
        std::vector<pbf::vec2f> positions;
        positions.reserve(count);

        for (int i = 0; i < count; ++i) {
            positions.emplace_back(dist(rng), dist(rng));
        }

        return positions;
    }

    // Create grid particle positions
    std::vector<pbf::vec2f> createGridParticles(int nx, int ny, float spacing) {
        std::vector<pbf::vec2f> positions;
        positions.reserve(nx * ny);

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                positions.emplace_back(i * spacing, j * spacing);
            }
        }

        return positions;
    }
};

TEST_F(SpatialHashTest, EmptyDomain) {
    std::vector<pbf::vec2f> positions;
    pbf::vec2f min_bounds(0.0f, 0.0f);
    pbf::vec2f max_bounds(domain_size, domain_size);
    pbf::SpatialHash<2> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    std::vector<int> particle_neighbors;
    spatial_hash.getNeighbors(0, particle_neighbors);
    EXPECT_TRUE(particle_neighbors.empty());
    std::vector<std::vector<int>> all_neighbors;
    spatial_hash.getAllNeighbors(all_neighbors);
    EXPECT_TRUE(all_neighbors.empty());
}

TEST_F(SpatialHashTest, SingleParticle) {
    std::vector<pbf::vec2f> positions = {{5.0f, 5.0f}};
    pbf::vec2f min_bounds(0.0f, 0.0f);
    pbf::vec2f max_bounds(domain_size, domain_size);
    pbf::SpatialHash<2> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    std::vector<int> particle_neighbors;
    spatial_hash.getNeighbors(0, particle_neighbors);
    EXPECT_TRUE(particle_neighbors.empty());
    std::vector<std::vector<int>> all_neighbors;
    spatial_hash.getAllNeighbors(all_neighbors);
    EXPECT_EQ(all_neighbors.size(), 1u);
    EXPECT_TRUE(all_neighbors[0].empty());
}

TEST_F(SpatialHashTest, PointQueryMatchesParticleNeighbors) {
    std::vector<pbf::vec2f> positions = {{5.0f, 5.0f}, {5.4f, 5.0f}, {7.0f, 5.0f}};
    pbf::vec2f min_bounds(0.0f, 0.0f);
    pbf::vec2f max_bounds(domain_size, domain_size);
    pbf::SpatialHash<2> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    std::vector<int> neighbors_from_query;
    spatial_hash.getNeighborsForPosition(positions[0], neighbors_from_query);
    std::vector<int> expected_neighbors{0, 1};

    EXPECT_EQ(neighbors_from_query, expected_neighbors);
}

TEST_F(SpatialHashTest, TwoParticlesWithinRange) {
    std::vector<pbf::vec2f> positions = {{5.0f, 5.0f}, {5.5f, 5.0f}};
    pbf::vec2f min_bounds(0.0f, 0.0f);
    pbf::vec2f max_bounds(domain_size, domain_size);
    pbf::SpatialHash<2> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    std::vector<int> neighbors_0;
    std::vector<int> neighbors_1;
    spatial_hash.getNeighbors(0, neighbors_0);
    spatial_hash.getNeighbors(1, neighbors_1);

    EXPECT_EQ(neighbors_0.size(), 1);
    EXPECT_EQ(neighbors_0[0], 1);
    EXPECT_EQ(neighbors_1.size(), 1);
    EXPECT_EQ(neighbors_1[0], 0);
}

TEST_F(SpatialHashTest, TwoParticlesOutOfRange) {
    std::vector<pbf::vec2f> positions = {{5.0f, 5.0f}, {7.0f, 5.0f}};
    pbf::vec2f min_bounds(0.0f, 0.0f);
    pbf::vec2f max_bounds(domain_size, domain_size);
    pbf::SpatialHash<2> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    std::vector<int> neighbors_0;
    std::vector<int> neighbors_1;
    spatial_hash.getNeighbors(0, neighbors_0);
    spatial_hash.getNeighbors(1, neighbors_1);

    EXPECT_TRUE(neighbors_0.empty());
    EXPECT_TRUE(neighbors_1.empty());
}

TEST_F(SpatialHashTest, GridParticlesCorrectness) {
    int nx = 5, ny = 5;
    float spacing = 0.8f;
    std::vector<pbf::vec2f> positions = createGridParticles(nx, ny, spacing);
    pbf::vec2f min_bounds(0.0f, 0.0f);
    pbf::vec2f max_bounds(domain_size, domain_size);
    pbf::SpatialHash<2> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    // Test all particles
    for (size_t i = 0; i < positions.size(); ++i) {
        std::vector<int> spatial_neighbors;
        spatial_hash.getNeighbors(static_cast<int>(i), spatial_neighbors);
        auto brute_neighbors = getNeighborsBruteForce2D(static_cast<int>(i), positions, h);

        // Sort both neighbor lists for comparison
        std::sort(spatial_neighbors.begin(), spatial_neighbors.end());
        std::sort(brute_neighbors.begin(), brute_neighbors.end());

        EXPECT_EQ(spatial_neighbors, brute_neighbors)
            << "Particle " << i << " has different neighbors";
    }
}

TEST_F(SpatialHashTest, RandomParticlesCorrectness) {
    int num_particles = 100;
    std::vector<pbf::vec2f> positions = createRandomParticles(num_particles);
    pbf::vec2f min_bounds(0.0f, 0.0f);
    pbf::vec2f max_bounds(domain_size, domain_size);
    pbf::SpatialHash<2> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    // Test all particles
    for (int i = 0; i < num_particles; ++i) {
        std::vector<int> spatial_neighbors;
        spatial_hash.getNeighbors(i, spatial_neighbors);
        auto brute_neighbors = getNeighborsBruteForce2D(i, positions, h);

        // Sort both neighbor lists for comparison
        std::sort(spatial_neighbors.begin(), spatial_neighbors.end());
        std::sort(brute_neighbors.begin(), brute_neighbors.end());

        EXPECT_EQ(spatial_neighbors, brute_neighbors)
            << "Particle " << i << " has different neighbors";
    }
}

TEST_F(SpatialHashTest, BoundaryConditions) {
    // Test particles near domain boundaries
    std::vector<pbf::vec2f> positions = {
        {0.1f, 0.1f},    // Near corner
        {0.1f, 9.9f},    // Near edge
        {9.9f, 0.1f},    // Near edge
        {9.9f, 9.9f},    // Near corner
        {5.0f, 5.0f}     // Center
    };

    pbf::vec2f min_bounds(0.0f, 0.0f);
    pbf::vec2f max_bounds(domain_size, domain_size);
    pbf::SpatialHash<2> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    // Test all particles
    for (size_t i = 0; i < positions.size(); ++i) {
        std::vector<int> spatial_neighbors;
        spatial_hash.getNeighbors(static_cast<int>(i), spatial_neighbors);
        auto brute_neighbors = getNeighborsBruteForce2D(static_cast<int>(i), positions, h);

        // Sort both neighbor lists for comparison
        std::sort(spatial_neighbors.begin(), spatial_neighbors.end());
        std::sort(brute_neighbors.begin(), brute_neighbors.end());

        EXPECT_EQ(spatial_neighbors, brute_neighbors)
            << "Particle " << i << " has different neighbors near boundary";
    }
}

TEST_F(SpatialHashTest, UpdatePositions) {
    std::vector<pbf::vec2f> positions1 = {{5.0f, 5.0f}, {5.5f, 5.0f}};
    std::vector<pbf::vec2f> positions2 = {{5.0f, 5.0f}, {7.0f, 5.0f}};

    pbf::vec2f min_bounds(0.0f, 0.0f);
    pbf::vec2f max_bounds(domain_size, domain_size);
    pbf::SpatialHash<2> spatial_hash(min_bounds, max_bounds, h);

    // First update
    spatial_hash.update(positions1);
    std::vector<int> neighbors_1;
    spatial_hash.getNeighbors(0, neighbors_1);
    EXPECT_EQ(neighbors_1.size(), 1);
    EXPECT_EQ(neighbors_1[0], 1);

    // Second update with different positions
    spatial_hash.update(positions2);
    std::vector<int> neighbors_2;
    spatial_hash.getNeighbors(0, neighbors_2);
    EXPECT_TRUE(neighbors_2.empty());
}

TEST_F(SpatialHashTest, GetAllNeighbors) {
    std::vector<pbf::vec2f> positions = createRandomParticles(50);
    pbf::vec2f min_bounds(0.0f, 0.0f);
    pbf::vec2f max_bounds(domain_size, domain_size);
    pbf::SpatialHash<2> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    std::vector<std::vector<int>> all_spatial;
    spatial_hash.getAllNeighbors(all_spatial);
    EXPECT_EQ(all_spatial.size(), positions.size());

    // Compare with individual queries
    for (size_t i = 0; i < positions.size(); ++i) {
        std::vector<int> individual;
        spatial_hash.getNeighbors(static_cast<int>(i), individual);
        EXPECT_EQ(all_spatial[i], individual);
    }
}

struct SpatialHash3DTest : ::testing::Test {
    std::mt19937 rng{1337};
    float h = 1.0f;
    float domain_size = 6.0f;

    std::vector<pbf::vec3f> createRandomParticles(int count) {
        std::uniform_real_distribution<float> dist(0.0f, domain_size);
        std::vector<pbf::vec3f> positions;
        positions.reserve(count);

        for (int i = 0; i < count; ++i) {
            positions.emplace_back(dist(rng), dist(rng), dist(rng));
        }

        return positions;
    }

    std::vector<pbf::vec3f> createGridParticles(int nx, int ny, int nz, float spacing) {
        std::vector<pbf::vec3f> positions;
        positions.reserve(nx * ny * nz);

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    positions.emplace_back(i * spacing, j * spacing, k * spacing);
                }
            }
        }

        return positions;
    }
};

TEST_F(SpatialHash3DTest, EmptyDomain) {
    std::vector<pbf::vec3f> positions;
    pbf::vec3f min_bounds(0.0f, 0.0f, 0.0f);
    pbf::vec3f max_bounds(domain_size, domain_size, domain_size);
    pbf::SpatialHash<3> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    std::vector<int> particle_neighbors;
    spatial_hash.getNeighbors(0, particle_neighbors);
    EXPECT_TRUE(particle_neighbors.empty());
    std::vector<std::vector<int>> all_neighbors;
    spatial_hash.getAllNeighbors(all_neighbors);
    EXPECT_TRUE(all_neighbors.empty());
}

TEST_F(SpatialHash3DTest, SingleParticle) {
    std::vector<pbf::vec3f> positions = {{2.0f, 2.0f, 2.0f}};
    pbf::vec3f min_bounds(0.0f, 0.0f, 0.0f);
    pbf::vec3f max_bounds(domain_size, domain_size, domain_size);
    pbf::SpatialHash<3> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    std::vector<int> particle_neighbors;
    spatial_hash.getNeighbors(0, particle_neighbors);
    EXPECT_TRUE(particle_neighbors.empty());
    std::vector<std::vector<int>> all_neighbors;
    spatial_hash.getAllNeighbors(all_neighbors);
    EXPECT_EQ(all_neighbors.size(), 1);
    EXPECT_TRUE(all_neighbors[0].empty());
}

TEST_F(SpatialHash3DTest, PointQueryMatchesParticleNeighbors) {
    std::vector<pbf::vec3f> positions = {{2.0f, 2.0f, 2.0f}, {2.6f, 2.0f, 2.0f}, {4.0f, 2.0f, 2.0f}};
    pbf::vec3f min_bounds(0.0f, 0.0f, 0.0f);
    pbf::vec3f max_bounds(domain_size, domain_size, domain_size);
    pbf::SpatialHash<3> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    std::vector<int> neighbors_from_query;
    spatial_hash.getNeighborsForPosition(positions[0], neighbors_from_query);
    std::vector<int> expected_neighbors{0, 1};

    EXPECT_EQ(neighbors_from_query, expected_neighbors);
}

TEST_F(SpatialHash3DTest, TwoParticlesWithinRange) {
    std::vector<pbf::vec3f> positions = {{2.0f, 2.0f, 2.0f}, {2.5f, 2.0f, 2.0f}};
    pbf::vec3f min_bounds(0.0f, 0.0f, 0.0f);
    pbf::vec3f max_bounds(domain_size, domain_size, domain_size);
    pbf::SpatialHash<3> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    std::vector<int> neighbors_0;
    std::vector<int> neighbors_1;
    spatial_hash.getNeighbors(0, neighbors_0);
    spatial_hash.getNeighbors(1, neighbors_1);

    EXPECT_EQ(neighbors_0.size(), 1);
    EXPECT_EQ(neighbors_0[0], 1);
    EXPECT_EQ(neighbors_1.size(), 1);
    EXPECT_EQ(neighbors_1[0], 0);
}

TEST_F(SpatialHash3DTest, TwoParticlesOutOfRange) {
    std::vector<pbf::vec3f> positions = {{2.0f, 2.0f, 2.0f}, {4.0f, 2.0f, 2.0f}};
    pbf::vec3f min_bounds(0.0f, 0.0f, 0.0f);
    pbf::vec3f max_bounds(domain_size, domain_size, domain_size);
    pbf::SpatialHash<3> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    std::vector<int> neighbors_0;
    std::vector<int> neighbors_1;
    spatial_hash.getNeighbors(0, neighbors_0);
    spatial_hash.getNeighbors(1, neighbors_1);

    EXPECT_TRUE(neighbors_0.empty());
    EXPECT_TRUE(neighbors_1.empty());
}

TEST_F(SpatialHash3DTest, GridParticlesCorrectness) {
    int nx = 4, ny = 4, nz = 4;
    float spacing = 0.8f;
    std::vector<pbf::vec3f> positions = createGridParticles(nx, ny, nz, spacing);
    pbf::vec3f min_bounds(0.0f, 0.0f, 0.0f);
    pbf::vec3f max_bounds(domain_size, domain_size, domain_size);
    pbf::SpatialHash<3> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    for (size_t i = 0; i < positions.size(); ++i) {
        std::vector<int> spatial_neighbors;
        spatial_hash.getNeighbors(static_cast<int>(i), spatial_neighbors);
        auto brute_neighbors = getNeighborsBruteForce3D(static_cast<int>(i), positions, h);

        std::sort(spatial_neighbors.begin(), spatial_neighbors.end());
        std::sort(brute_neighbors.begin(), brute_neighbors.end());

        EXPECT_EQ(spatial_neighbors, brute_neighbors)
            << "3D particle " << i << " has different neighbors";
    }
}

TEST_F(SpatialHash3DTest, RandomParticlesCorrectness) {
    int num_particles = 75;
    std::vector<pbf::vec3f> positions = createRandomParticles(num_particles);
    pbf::vec3f min_bounds(0.0f, 0.0f, 0.0f);
    pbf::vec3f max_bounds(domain_size, domain_size, domain_size);
    pbf::SpatialHash<3> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    for (int i = 0; i < num_particles; ++i) {
        std::vector<int> spatial_neighbors;
        spatial_hash.getNeighbors(i, spatial_neighbors);
        auto brute_neighbors = getNeighborsBruteForce3D(i, positions, h);

        std::sort(spatial_neighbors.begin(), spatial_neighbors.end());
        std::sort(brute_neighbors.begin(), brute_neighbors.end());

        EXPECT_EQ(spatial_neighbors, brute_neighbors)
            << "3D particle " << i << " has different neighbors";
    }
}

TEST_F(SpatialHash3DTest, BoundaryConditions) {
    std::vector<pbf::vec3f> positions = {
        {0.1f, 0.1f, 0.1f},
        {0.1f, 0.1f, domain_size - 0.1f},
        {0.1f, domain_size - 0.1f, 0.1f},
        {domain_size - 0.1f, 0.1f, 0.1f},
        {domain_size - 0.1f, domain_size - 0.1f, domain_size - 0.1f},
        {3.0f, 3.0f, 3.0f}
    };

    pbf::vec3f min_bounds(0.0f, 0.0f, 0.0f);
    pbf::vec3f max_bounds(domain_size, domain_size, domain_size);
    pbf::SpatialHash<3> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    for (size_t i = 0; i < positions.size(); ++i) {
        std::vector<int> spatial_neighbors;
        spatial_hash.getNeighbors(static_cast<int>(i), spatial_neighbors);
        auto brute_neighbors = getNeighborsBruteForce3D(static_cast<int>(i), positions, h);

        std::sort(spatial_neighbors.begin(), spatial_neighbors.end());
        std::sort(brute_neighbors.begin(), brute_neighbors.end());

        EXPECT_EQ(spatial_neighbors, brute_neighbors)
            << "3D particle " << i << " has different neighbors near boundary";
    }
}

TEST_F(SpatialHash3DTest, UpdatePositions) {
    std::vector<pbf::vec3f> positions1 = {{2.0f, 2.0f, 2.0f}, {2.5f, 2.0f, 2.0f}};
    std::vector<pbf::vec3f> positions2 = {{2.0f, 2.0f, 2.0f}, {4.0f, 2.0f, 2.0f}};

    pbf::vec3f min_bounds(0.0f, 0.0f, 0.0f);
    pbf::vec3f max_bounds(domain_size, domain_size, domain_size);
    pbf::SpatialHash<3> spatial_hash(min_bounds, max_bounds, h);

    spatial_hash.update(positions1);
    std::vector<int> neighbors_1;
    spatial_hash.getNeighbors(0, neighbors_1);
    EXPECT_EQ(neighbors_1.size(), 1);
    EXPECT_EQ(neighbors_1[0], 1);

    spatial_hash.update(positions2);
    std::vector<int> neighbors_2;
    spatial_hash.getNeighbors(0, neighbors_2);
    EXPECT_TRUE(neighbors_2.empty());
}

TEST_F(SpatialHash3DTest, GetAllNeighbors) {
    std::vector<pbf::vec3f> positions = createRandomParticles(40);
    pbf::vec3f min_bounds(0.0f, 0.0f, 0.0f);
    pbf::vec3f max_bounds(domain_size, domain_size, domain_size);
    pbf::SpatialHash<3> spatial_hash(min_bounds, max_bounds, h);
    spatial_hash.update(positions);

    std::vector<std::vector<int>> all_spatial;
    spatial_hash.getAllNeighbors(all_spatial);
    EXPECT_EQ(all_spatial.size(), positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        std::vector<int> individual;
        spatial_hash.getNeighbors(static_cast<int>(i), individual);
        EXPECT_EQ(all_spatial[i], individual);
    }
}

} // namespace