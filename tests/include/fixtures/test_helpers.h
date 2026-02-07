#pragma once
#include <array>
#include <random>
#include <vector>
#include "pbf/vec.h"

namespace testing {

    // Brute force returns all neighbors around particle self_index closer than h
    // This does not return self-adjacency
    template <int Dim>
    inline std::vector<int> get_neighbors_slow(int self_index, std::vector<pbf::Vec<Dim>> const & positions, float h) {
        std::vector<int> neighbors;
        pbf::Vec<Dim> position_i = positions.at(self_index);
        for (int j = 0; j < positions.size(); ++j) {
            if (self_index == j)
                continue;
            pbf::Vec<Dim> position_j = positions.at(j);
            pbf::Vec<Dim> diff = position_i - position_j;
            float distance = diff.length();
            if (distance <= h)
                neighbors.push_back(j);
        }
        return neighbors;
    }

    template <int Dim, class URBG>
    inline std::vector<pbf::Vec<Dim>>
    jittered_grid(std::array<int, Dim> const & dims, float dx, float jitter, URBG& rng)
    {
        static_assert(Dim == 2 || Dim == 3, "jittered_grid supports only 2D or 3D.");
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);

        int total = 1;
        for (int d = 0; d < Dim; ++d) {
            total *= dims[d];
        }

        std::vector<pbf::Vec<Dim>> pos;
        pos.reserve(total);

        if constexpr (Dim == 2) {
            for (int j = 0; j < dims[1]; ++j)
            {
                for (int i = 0; i < dims[0]; ++i)
                {
                    pbf::Vec<Dim> p{i * dx, j * dx};
                    p.x += (uni(rng) - 0.5f) * dx * jitter;
                    p.y += (uni(rng) - 0.5f) * dx * jitter;
                    pos.push_back(p);
                }
            }
        } else {
            for (int k = 0; k < dims[2]; ++k)
            {
                for (int j = 0; j < dims[1]; ++j)
                {
                    for (int i = 0; i < dims[0]; ++i)
                    {
                        pbf::Vec<Dim> p{i * dx, j * dx, k * dx};
                        p.x += (uni(rng) - 0.5f) * dx * jitter;
                        p.y += (uni(rng) - 0.5f) * dx * jitter;
                        p.z += (uni(rng) - 0.5f) * dx * jitter;
                        pos.push_back(p);
                    }
                }
            }
        }

        return pos;
    }
}
