#include <gtest/gtest.h>
#include <random>
#include <chrono>
#include "pbf/spatial_hash.h"
#include "fixtures/test_helpers.h"

namespace {

// Helper function for brute force neighbor search
std::vector<int> getNeighborsBruteForce(int self_index,
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
    pbf::SpatialHash spatial_hash(0, 0, domain_size, domain_size, h);
    spatial_hash.update(positions);

    EXPECT_TRUE(spatial_hash.getNeighbors(0).empty());
    auto all_neighbors = spatial_hash.getAllNeighbors();
    EXPECT_TRUE(all_neighbors.empty());
}

TEST_F(SpatialHashTest, SingleParticle) {
    std::vector<pbf::vec2f> positions = {{5.0f, 5.0f}};
    pbf::SpatialHash spatial_hash(0, 0, domain_size, domain_size, h);
    spatial_hash.update(positions);

    EXPECT_TRUE(spatial_hash.getNeighbors(0).empty());
    auto all_neighbors = spatial_hash.getAllNeighbors();
    EXPECT_EQ(all_neighbors.size(), 1);
    EXPECT_TRUE(all_neighbors[0].empty());
}

TEST_F(SpatialHashTest, TwoParticlesWithinRange) {
    std::vector<pbf::vec2f> positions = {{5.0f, 5.0f}, {5.5f, 5.0f}};
    pbf::SpatialHash spatial_hash(0, 0, domain_size, domain_size, h);
    spatial_hash.update(positions);

    auto neighbors_0 = spatial_hash.getNeighbors(0);
    auto neighbors_1 = spatial_hash.getNeighbors(1);

    EXPECT_EQ(neighbors_0.size(), 1);
    EXPECT_EQ(neighbors_0[0], 1);
    EXPECT_EQ(neighbors_1.size(), 1);
    EXPECT_EQ(neighbors_1[0], 0);
}

TEST_F(SpatialHashTest, TwoParticlesOutOfRange) {
    std::vector<pbf::vec2f> positions = {{5.0f, 5.0f}, {7.0f, 5.0f}};
    pbf::SpatialHash spatial_hash(0, 0, domain_size, domain_size, h);
    spatial_hash.update(positions);

    auto neighbors_0 = spatial_hash.getNeighbors(0);
    auto neighbors_1 = spatial_hash.getNeighbors(1);

    EXPECT_TRUE(neighbors_0.empty());
    EXPECT_TRUE(neighbors_1.empty());
}

TEST_F(SpatialHashTest, GridParticlesCorrectness) {
    int nx = 5, ny = 5;
    float spacing = 0.8f;
    std::vector<pbf::vec2f> positions = createGridParticles(nx, ny, spacing);
    pbf::SpatialHash spatial_hash(0, 0, domain_size, domain_size, h);
    spatial_hash.update(positions);

    // Test all particles
    for (size_t i = 0; i < positions.size(); ++i) {
        auto spatial_neighbors = spatial_hash.getNeighbors(static_cast<int>(i));
        auto brute_neighbors = getNeighborsBruteForce(static_cast<int>(i), positions, h);

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
    pbf::SpatialHash spatial_hash(0, 0, domain_size, domain_size, h);
    spatial_hash.update(positions);

    // Test all particles
    for (int i = 0; i < num_particles; ++i) {
        auto spatial_neighbors = spatial_hash.getNeighbors(i);
        auto brute_neighbors = getNeighborsBruteForce(i, positions, h);

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

    pbf::SpatialHash spatial_hash(0, 0, domain_size, domain_size, h);
    spatial_hash.update(positions);

    // Test all particles
    for (size_t i = 0; i < positions.size(); ++i) {
        auto spatial_neighbors = spatial_hash.getNeighbors(static_cast<int>(i));
        auto brute_neighbors = getNeighborsBruteForce(static_cast<int>(i), positions, h);

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

    pbf::SpatialHash spatial_hash(0, 0, domain_size, domain_size, h);

    // First update
    spatial_hash.update(positions1);
    auto neighbors_1 = spatial_hash.getNeighbors(0);
    EXPECT_EQ(neighbors_1.size(), 1);
    EXPECT_EQ(neighbors_1[0], 1);

    // Second update with different positions
    spatial_hash.update(positions2);
    auto neighbors_2 = spatial_hash.getNeighbors(0);
    EXPECT_TRUE(neighbors_2.empty());
}

TEST_F(SpatialHashTest, GetAllNeighbors) {
    std::vector<pbf::vec2f> positions = createRandomParticles(50);
    pbf::SpatialHash spatial_hash(0, 0, domain_size, domain_size, h);
    spatial_hash.update(positions);

    auto all_spatial = spatial_hash.getAllNeighbors();
    EXPECT_EQ(all_spatial.size(), positions.size());

    // Compare with individual queries
    for (size_t i = 0; i < positions.size(); ++i) {
        auto individual = spatial_hash.getNeighbors(static_cast<int>(i));
        EXPECT_EQ(all_spatial[i], individual);
    }
}

// TEST_F(SpatialHashTest, PerformanceComparison) {
//     int num_particles = 1000;
//     std::vector<pbf::vec2f> positions = createRandomParticles(num_particles);
//     pbf::SpatialHash spatial_hash(0, 0, domain_size, domain_size, h);
//     spatial_hash.update(positions);

//     // Time spatial hash
//     auto start = std::chrono::high_resolution_clock::now();
//     auto spatial_neighbors = spatial_hash.getAllNeighbors();
//     auto end = std::chrono::high_resolution_clock::now();
//     auto spatial_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

//     // Time brute force
//     start = std::chrono::high_resolution_clock::now();
//     std::vector<std::vector<int>> brute_neighbors;
//     brute_neighbors.reserve(num_particles);
//     for (int i = 0; i < num_particles; ++i) {
//         brute_neighbors.push_back(getNeighborsBruteForce(i, positions, h));
//     }
//     end = std::chrono::high_resolution_clock::now();
//     auto brute_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

//     // Verify correctness
//     for (size_t i = 0; i < positions.size(); ++i) {
//         std::sort(spatial_neighbors[i].begin(), spatial_neighbors[i].end());
//         std::sort(brute_neighbors[i].begin(), brute_neighbors[i].end());
//         EXPECT_EQ(spatial_neighbors[i], brute_neighbors[i]);
//     }

//     // Spatial hash should be faster for large particle counts
//     // (Note: This might not always be true for small particle counts due to overhead)
//     std::cout << "Spatial hash time: " << spatial_time.count() << " microseconds\n";
//     std::cout << "Brute force time: " << brute_time.count() << " microseconds\n";
//     std::cout << "Speedup: " << static_cast<double>(brute_time.count()) / spatial_time.count() << "x\n";
// }


} // namespace