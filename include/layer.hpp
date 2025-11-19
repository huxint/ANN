#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include "util.hpp"

namespace nn {
    class Layer {
    public:
        virtual ~Layer() = default;
        virtual Vector forward(const Vector &x) = 0;
        virtual Vector backward(const Vector &grad, double lr) = 0;
    };

    class DenseLayer : public Layer {
    public:
        DenseLayer(int input_size, int output_size, Activation activation = Activation::Sigmoid)
        : activation_(activation) { // 初始化权重和偏置
            std::mt19937 gen(std::random_device{}());
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
            return apply_activation(last_Z_, activation_);
        }

        Vector backward(const Vector &grad, double lr) override {
            // 1. 激活函数的链式法则：从 dL/da -> dL/dz
            Vector grad_z = grad.cwiseProduct(activation_derivative(last_Z_, activation_));

            Matrix dW = grad_z * last_input_.transpose(); // dL/dW
            Vector db = grad_z;                           // dL/db

            // 3. 传给前一层的梯度 dL/dx
            Vector grad_input = W_.transpose() * grad_z;

            // 4. SGD 更新参数
            W_ -= lr * dW;
            b_ -= lr * db;
            return grad_input;
        }

    private:
        Matrix W_; // 权重矩阵
        Vector b_; // 偏置向量

        Vector last_input_; // 最后一次输入 x
        Vector last_Z_;     // Z = W * x + b
        Activation activation_;
    };
} // namespace nn
