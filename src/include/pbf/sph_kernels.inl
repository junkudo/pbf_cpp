namespace pbf::sph {
    inline Poly6<2>::Poly6(float h)
        : h_(h)
        , h2_(h * h)
        , coeff_(4.0f / (std::numbers::pi_v<float> * h2_ * h2_ * h2_ * h2_))
        , eval_at_zero_(coeff_ * h2_ * h2_ * h2_) {}

    inline float Poly6<2>::eval(float r) const {
        if (r > h_) {
            return 0.0f;
        }
        const float r2 = r * r;
        const float diff = h2_ - r2;
        return coeff_ * diff * diff * diff;
    }

    inline float Poly6<2>::evalAtZero() const {
        return eval_at_zero_;
    }

    inline float Poly6<2>::dWdr(float r) const {
        if (r > h_) {
            return 0.0f;
        }
        const float r2 = r * r;
        const float diff = h2_ - r2;
        return -6.0f * coeff_ * r * diff * diff;
    }

    inline Poly6<3>::Poly6(float h)
        : h_(h)
        , h2_(h * h)
        , coeff_(315.0f / (64.0f * std::numbers::pi_v<float> * h2_ * h2_ * h2_ * h2_ * h))
        , eval_at_zero_(coeff_ * h2_ * h2_ * h2_) {}

    inline float Poly6<3>::eval(float r) const {
        if (r > h_) {
            return 0.0f;
        }
        const float r2 = r * r;
        const float diff = h2_ - r2;
        return coeff_ * diff * diff * diff;
    }

    inline float Poly6<3>::evalAtZero() const {
        return eval_at_zero_;
    }

    inline float Poly6<3>::dWdr(float r) const {
        if (r > h_) {
            return 0.0f;
        }
        const float r2 = r * r;
        const float diff = h2_ - r2;
        return -6.0f * coeff_ * r * diff * diff;
    }

    inline Spikey<2>::Spikey(float h)
        : h_(h)
        , coeff_(10.0f / (std::numbers::pi_v<float> * h * h * h * h * h))
        , eval_at_zero_(coeff_ * h * h * h) {}

    inline float Spikey<2>::eval(float r) const {
        if (r > h_) {
            return 0.0f;
        }
        const float diff = h_ - r;
        return coeff_ * diff * diff * diff;
    }

    inline float Spikey<2>::evalAtZero() const {
        return eval_at_zero_;
    }

    inline float Spikey<2>::dWdr(float r) const {
        if (r > h_ || r <= 0.0f) {
            return 0.0f;
        }
        const float diff = h_ - r;
        return -3.0f * coeff_ * diff * diff;
    }

    inline Spikey<3>::Spikey(float h)
        : h_(h)
        , coeff_(15.0f / (std::numbers::pi_v<float> * h * h * h * h * h * h))
        , eval_at_zero_(coeff_ * h * h * h) {}

    inline float Spikey<3>::eval(float r) const {
        if (r > h_) {
            return 0.0f;
        }
        const float diff = h_ - r;
        return coeff_ * diff * diff * diff;
    }

    inline float Spikey<3>::evalAtZero() const {
        return eval_at_zero_;
    }

    inline float Spikey<3>::dWdr(float r) const {
        if (r > h_ || r <= 0.0f) {
            return 0.0f;
        }
        const float diff = h_ - r;
        return -3.0f * coeff_ * diff * diff;
    }

    inline CubicSpline<2>::CubicSpline(float h)
        : h_(h)
        , inv_h_(1.0f / h)
        , inv_h2_(inv_h_ * inv_h_)
        , k_(40.0f / (7.0f * std::numbers::pi_v<float> * h * h)) {}

    inline float CubicSpline<2>::eval(float r) const {
        const float q = r * inv_h_;
        if (q > 1.0f) {
            return 0.0f;
        }
        if (q <= 0.5f) {
            const float q2 = q * q;
            const float q3 = q2 * q;
            return k_ * (6.0f * q3 - 6.0f * q2 + 1.0f);
        }
        const float factor = 1.0f - q;
        return k_ * (2.0f * factor * factor * factor);
    }

    inline float CubicSpline<2>::evalAtZero() const {
        return k_;
    }

    inline float CubicSpline<2>::dWdr(float r) const {
        const float q = r * inv_h_;
        if (r <= 1.0e-6f || q > 1.0f) {
            return 0.0f;
        }
        if (q <= 0.5f) {
            const float q2 = q * q;
            return k_ * (18.0f * q2 - 12.0f * q) * inv_h_;
        }
        const float factor = 1.0f - q;
        return k_ * (-6.0f * factor * factor) * inv_h_;
    }

    inline CubicSpline<3>::CubicSpline(float h)
        : h_(h)
        , inv_h_(1.0f / h)
        , k_(8.0f / (std::numbers::pi_v<float> * h * h * h)) {}

    inline float CubicSpline<3>::eval(float r) const {
        const float q = r * inv_h_;
        if (q > 1.0f) {
            return 0.0f;
        }
        if (q <= 0.5f) {
            const float q2 = q * q;
            const float q3 = q2 * q;
            return k_ * (6.0f * q3 - 6.0f * q2 + 1.0f);
        }
        const float factor = 1.0f - q;
        return k_ * (2.0f * factor * factor * factor);
    }

    inline float CubicSpline<3>::evalAtZero() const {
        return k_;
    }

    inline float CubicSpline<3>::dWdr(float r) const {
        const float q = r * inv_h_;
        if (r <= 1.0e-6f || q > 1.0f) {
            return 0.0f;
        }
        if (q <= 0.5f) {
            const float q2 = q * q;
            return k_ * (18.0f * q2 - 12.0f * q) * inv_h_;
        }
        const float factor = 1.0f - q;
        return k_ * (-6.0f * factor * factor) * inv_h_;
    }

    template <int Dim, typename Kernel>
    float computeDensity(int self_index, float mass, float h,
        std::vector<int> const & neighbors,
        std::vector<Vec<Dim>> const & positions) {
            const Kernel kernel(h);
            // Calculate self contribution
            float density = mass * kernel.evalAtZero();
            Vec<Dim> const & pos_i = positions[self_index];
            for (int neighbor : neighbors) {
                // Assumes neighbors contains only fluid particles; boundary density is handled separately.
                Vec<Dim> const & pos_j = positions[neighbor];
                density += mass * kernel.eval((pos_i - pos_j).length());
            }

            return density;

    }

    template <int Dim, typename Kernel>
    float computeBoundaryPsi(int self_index,
        float h,
        float rest_density,
        float pressure_scale,
        std::vector<int> const& boundary_neighbors,
        std::vector<Vec<Dim>> const& boundary_positions) {
            const Kernel kernel(h);
            Vec<Dim> const& pi = boundary_positions[self_index];
            float delta = kernel.evalAtZero();

            for (int neighbor_index : boundary_neighbors) {
                if (neighbor_index == self_index) {
                    continue;
                }
                Vec<Dim> const& pj = boundary_positions[neighbor_index];
                delta += kernel.eval((pi - pj).length());
            }

            const float volume = 1.0f / delta;
            return pressure_scale * rest_density * volume;
    }

    template <int Dim, typename Kernel>
    void computeXsphViscosity(float viscosity, float mass, float h,
        std::vector<Vec<Dim>> const& positions,
        std::vector<std::vector<int>> const& neighbors,
        std::vector<Vec<Dim>>& velocities) {
        const Kernel kernel(h);
        const int num_particles = static_cast<int>(positions.size());
        std::vector<float> densities(num_particles, 0.0f);
        for (int i = 0; i < num_particles; ++i) {
            densities[i] = computeDensity<Dim, Kernel>(i, mass, h, neighbors[i], positions);
        }

        std::vector<Vec<Dim>> deltas(num_particles, Vec<Dim>::zero());
        for (int i = 0; i < num_particles; ++i) {
            const auto& vi = velocities[i];
            const auto& xi = positions[i];
            for (int neighbor : neighbors[i]) {
                const float density_j = densities[neighbor];
                if (density_j <= 0.0f) {
                    continue;
                }
                const auto& vj = velocities[neighbor];
                const auto& xj = positions[neighbor];
                deltas[i] += (vj - vi) * (mass / density_j) * kernel.eval((xi - xj).length());
            }
        }

        for (int i = 0; i < num_particles; ++i) {
            velocities[i] += viscosity * deltas[i];
        }
    }
}