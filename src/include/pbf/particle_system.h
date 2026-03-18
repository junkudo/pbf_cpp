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

    ParticleSystem(const std::vector<VecType>& positions, const PhysicsConfig& config)
        : num_particles_(static_cast<int>(positions.size())),
          config_(config),
          positions_(positions),
          velocities_(positions.size(), VecType::zero()),
          lambdas_(positions.size(), 0.0f) {}

    ParticleSystem(const std::array<int, Dim>& counts,
                   const PhysicsConfig& config,
                   const VecType& origin_offset = VecType::zero())
        : ParticleSystem(createJitteredGridPositions(counts, config, origin_offset), config) {}

    static std::vector<VecType> createJitteredGridPositions(const std::array<int, Dim>& counts,
                                                           const PhysicsConfig& config,
                                                           const VecType& origin_offset = VecType::zero()) {
        int num_particles = 1;
        for (int dim = 0; dim < Dim; ++dim) {
            num_particles *= counts[dim];
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> jitter(-config.jitterFactor * config.particleSpacing,
                                                     config.jitterFactor * config.particleSpacing);

        std::vector<VecType> positions;
        positions.resize(num_particles, VecType::zero());

        for (int i = 0; i < num_particles; ++i) {
            VecType position = VecType::zero();
            int remainder = i;
            for (int dim = 0; dim < Dim; ++dim) {
                const int index = remainder % counts[dim];
                remainder /= counts[dim];
                position[dim] = index * config.particleSpacing + jitter(gen) + origin_offset[dim];
            }
            positions[i] = position;
        }

        return positions;
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
    void predictPositions(std::vector<VecType>& predicted) const {
        predicted = positions_;
        for (int i = 0; i < num_particles_; ++i) {
            predicted[i] += velocities_[i] * config_.timeStep;
        }
    }

    // Apply position corrections and update velocities from full displacement.
    void updatePositions(const std::vector<VecType>& corrections) {
        for (int i = 0; i < num_particles_; ++i) {
            const VecType previous_position = positions_[i];
            positions_[i] += velocities_[i] * config_.timeStep + corrections[i];
            velocities_[i] = (positions_[i] - previous_position) / config_.timeStep;
        }
    }

    const std::vector<VecType>& getPositions() const { return positions_; }
    int getNumParticles() const { return num_particles_; }

private:
    static_assert(Dim > 0, "ParticleSystem requires a positive dimension");
};

}