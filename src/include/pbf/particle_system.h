#pragma once

#include <array>
#include <random>
#include <vector>

#include "pbf/config.h"
#include "pbf/vec.h"

namespace pbf {

// Lightweight particle system for Dim-dimensional PBF simulations.
template<int Dim>
class ParticleSystem {
public:
    using VecType = Vec<Dim>;

    // Create a jittered grid of particles with per-dimension counts.
    std::vector<VecType> positions_;
    std::vector<VecType> velocities_;
    std::vector<float> lambdas_;
    int num_particles_;
    PhysicsConfig config_;

    ParticleSystem(const std::array<int, Dim>& counts,
                   const PhysicsConfig& config,
                   const VecType& origin_offset = VecType::zero())
        : num_particles_(1), config_(config) {
        for (int dim = 0; dim < Dim; ++dim) {
            num_particles_ *= counts[dim];
        }

        positions_.resize(num_particles_);
        velocities_.resize(num_particles_);
        lambdas_.resize(num_particles_);

        initializeJitteredGrid(counts, origin_offset);
    }

    // Apply gravity in the +Y axis direction (index 1).
    void updateVelocityFromGravity() {
        if constexpr (Dim > 1) {
            for (auto& v : velocities_) {
                v[1] -= config_.gravity * config_.timeStep;
            }
        }
    }

    // Predict positions using current velocities and timestep.
    std::vector<VecType> predictPositions() const {
        std::vector<VecType> predicted = positions_;
        for (int i = 0; i < num_particles_; ++i) {
            predicted[i] += velocities_[i] * config_.timeStep;
        }
        return predicted;
    }

    // Apply position corrections and update velocities from them.
    void updatePositions(const std::vector<VecType>& corrections) {
        for (int i = 0; i < num_particles_; ++i) {
            positions_[i] += corrections[i];
            velocities_[i] = corrections[i] / config_.timeStep;
        }
    }

    const std::vector<VecType>& getPositions() const { return positions_; }
    int getNumParticles() const { return num_particles_; }

private:
    void initializeJitteredGrid(const std::array<int, Dim>& counts,
                                const VecType& origin_offset) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> jitter(-config_.jitterFactor * config_.particleSpacing,
                                                     config_.jitterFactor * config_.particleSpacing);

        for (int i = 0; i < num_particles_; ++i) {
            VecType position = VecType::zero();
            int remainder = i;
            for (int dim = 0; dim < Dim; ++dim) {
                const int index = remainder % counts[dim];
                remainder /= counts[dim];
                position[dim] = index * config_.particleSpacing + jitter(gen) + origin_offset[dim];
            }
            positions_[i] = position;
            velocities_[i] = VecType::zero();
        }
    }
};

}