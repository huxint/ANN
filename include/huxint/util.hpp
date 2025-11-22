#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <random>

namespace nn {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    enum class Activation { Sigmoid, ReLU, LeakyReLU, Tanh, None };

    inline constexpr double kLeakyReluSlope = 0.01;

    inline std::mt19937 &global_rng() {
        static std::mt19937 rng(std::random_device{}());
        return rng;
    }

    inline void set_global_seed(std::uint64_t seed) {
        global_rng().seed(seed);
    }

    inline Vector sigmoid(const Vector &x) {
        return (1.0 / (1.0 + (-x.array()).exp())).matrix();
    }

    inline Vector sigmoid_deriv_activated(const Vector &activate_x) {
        return (activate_x.array() * (1.0 - activate_x.array())).matrix();
    }

    inline Vector relu(const Vector &x) {
        return x.array().max(0.0).matrix();
    }

    inline Vector relu_deriv_activated(const Vector &activate_x) {
        return (activate_x.array() > 0.0).cast<double>().matrix();
    }

    inline Vector leaky_relu(const Vector &x, double alpha = kLeakyReluSlope) {
        return x.unaryExpr([alpha](double v) {
            return v > 0.0 ? v : alpha * v;
        });
    }

    inline Vector leaky_relu_deriv_activated(const Vector &activate_x, double alpha = kLeakyReluSlope) {
        return activate_x.unaryExpr([alpha](double v) {
            return v > 0.0 ? 1.0 : alpha;
        });
    }

    inline Vector tanh(const Vector &x) {
        return x.array().tanh().matrix();
    }

    inline Vector tanh_deriv_activated(const Vector &activate_x) {
        return (1.0 - activate_x.array().square()).matrix();
    }

    inline Vector apply_activation(const Vector &x, Activation activation) {
        switch (activation) {
            case Activation::ReLU:
                return relu(x);
            case Activation::LeakyReLU:
                return leaky_relu(x);
            case Activation::None:
                return x;
            case Activation::Tanh:
                return tanh(x);
            case Activation::Sigmoid:
            default:
                return sigmoid(x);
        }
    }

    inline Vector activation_derivative(const Vector &x, Activation activation) {
        switch (activation) {
            case Activation::ReLU:
                return relu_deriv_activated(x);
            case Activation::LeakyReLU:
                return leaky_relu_deriv_activated(x);
            case Activation::None:
                return Vector::Ones(x.size());
            case Activation::Tanh:
                return tanh_deriv_activated(x);
            case Activation::Sigmoid:
            default:
                return sigmoid_deriv_activated(x);
        }
    }
} // namespace nn
