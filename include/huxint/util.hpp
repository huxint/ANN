#pragma once
#include <Eigen/Dense>

namespace nn {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    enum class Activation { Sigmoid, ReLU, Tanh, None };

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
