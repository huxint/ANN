#pragma once
#include <Eigen/Dense>

namespace nn {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    enum class Activation { Sigmoid, ReLU, None };

    inline Vector sigmoid(const Vector &x) {
        return (1.0 / (1.0 + (-x.array()).exp())).matrix();
    }

    inline Vector sigmoid_deriv(const Vector &x) {
        Vector sig = sigmoid(x);
        return (sig.array() * (1.0 - sig.array())).matrix();
    }

    inline Vector relu(const Vector &x) {
        return x.array().max(0.0).matrix();
    }

    inline Vector relu_deriv(const Vector &x) {
        return (x.array() > 0.0).cast<double>().matrix();
    }

    inline Vector apply_activation(const Vector &x, Activation activation) {
        switch (activation) {
            case Activation::ReLU:
                return relu(x);
            case Activation::None:
                return x;
            case Activation::Sigmoid:
            default:
                return sigmoid(x);
        }
    }

    inline Vector activation_derivative(const Vector &x, Activation activation) {
        switch (activation) {
            case Activation::ReLU:
                return relu_deriv(x);
            case Activation::None:
                return Vector::Ones(x.size());
            case Activation::Sigmoid:
            default:
                return sigmoid_deriv(x);
        }
    }
} // namespace nn
