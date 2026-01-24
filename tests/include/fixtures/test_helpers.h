#pragma once
#include <vector>
#include "pbf/vec2f.h"

namespace testing {

    // Brute force returns all neighbors around particle self_index closer than h
    // This does not return self-adjacency
    inline std::vector<int> get_neighbors_slow(int self_index, std::vector<pbf::vec2f> const & positions, float h) {
        std::vector<int> neighbors;
        pbf::vec2f position_i = positions.at(self_index);
        for (int j = 0; j < positions.size(); ++j) {
            if (self_index == j)
                continue;
            pbf::vec2f position_j = positions.at(j);
            pbf::vec2f diff = position_i - position_j;
            float distance = diff.length();
            if (distance <= h)
                neighbors.push_back(j);
        }
        return neighbors;
    }

    template <class URBG>
    inline std::vector<pbf::vec2f>
    jittered_grid(int nx, int ny, float dx, float jitter, URBG& rng)
    {
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);

        std::vector<pbf::vec2f> pos;
        pos.reserve(nx * ny);

        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                pbf::vec2f p{i * dx, j * dx};
                p.x += (uni(rng) - 0.5f) * dx * jitter;
                p.y += (uni(rng) - 0.5f) * dx * jitter;
                pos.push_back(p);
            }
        }

        return pos;
    }
}
