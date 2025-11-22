#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>
#include "util.hpp"

namespace nn {
    class Layer {
    public:
        virtual ~Layer() = default;
        virtual Vector forward(const Vector &x) = 0;
        virtual Vector backward(const Vector &grad, double lr) = 0;
        virtual Vector predict(const Vector &x) {
            return forward(x);
        }
    };

    class DenseLayer : public Layer {
    public:
        DenseLayer(int input_size, int output_size, Activation activation = Activation::Sigmoid)
        : activation_(activation) { // 初始化权重和偏置
            auto &gen = global_rng();
            std::normal_distribution<double> dist(0.0, 1.0);

            b_ = Vector::Zero(output_size);

            double scale = 1.0 / std::sqrt(static_cast<double>(input_size));
            W_ = Matrix::NullaryExpr(output_size, input_size, [&]() {
                return dist(gen) * scale;
            });
        }

        Vector forward(const Vector &x) override {
            last_input_ = x;
            last_Z_ = W_ * x + b_;
            last_activation_ = apply_activation(last_Z_, activation_);
            return last_activation_;
        }

        Vector backward(const Vector &grad, double lr) override {
            // 1. 激活函数的链式法则：从 dL/da -> dL/dz
            Vector grad_z = grad.cwiseProduct(activation_derivative(last_activation_, activation_));

            Matrix dW = grad_z * last_input_.transpose(); // dL/dW
            Vector db = grad_z;                           // dL/db

            // 3. 传给前一层的梯度 dL/dx
            Vector grad_input = W_.transpose() * grad_z;

            // 4. SGD 更新参数
            W_ -= lr * dW;
            b_ -= lr * db;
            return grad_input;
        }

        Vector predict(const Vector &x) override {
            return apply_activation(W_ * x + b_, activation_);
        }

    private:
        Matrix W_; // 权重矩阵
        Vector b_; // 偏置向量

        Vector last_input_; // 最后一次输入 x
        Vector last_Z_;     // Z = W * x + b
        Activation activation_;
        Vector last_activation_; // 最后一次激活函数输出 a = f(Z)
    };

    class DropoutLayer : public Layer {
    public:
        explicit DropoutLayer(double dropout_rate)
        : rate_(std::clamp(dropout_rate, 0.0, 0.99)),
          scale_(rate_ < 1.0 ? 1.0 / (1.0 - rate_) : 0.0) {}

        Vector forward(const Vector &x) override {
            if (rate_ == 0.0) {
                return predict(x);
            }
            mask_.resize(x.size());
            auto &gen = global_rng();
            std::bernoulli_distribution dist(1.0 - rate_);
            for (Eigen::Index i = 0; i < mask_.size(); ++i) {
                mask_(i) = dist(gen) ? 1.0 : 0.0;
            }
            return (x.array() * mask_.array() * scale_).matrix();
        }

        Vector backward(const Vector &grad, double /*lr*/) override {
            if (mask_.size() == 0 || rate_ <= 0.0) {
                return grad;
            }
            return (grad.array() * mask_.array() * scale_).matrix();
        }

        Vector predict(const Vector &x) override {
            mask_ = Vector::Ones(x.size());
            return x;
        }

    private:
        double rate_;  // 失活概率
        double scale_; // 期望保持一致的缩放系数
        Vector mask_;
    };
} // namespace nn
