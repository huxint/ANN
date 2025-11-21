#pragma once
#include <Eigen/Dense>
#include <cstddef>
#include "network.hpp"
#include "util.hpp"

namespace nn {
    class Trainer {
    public:
        Trainer(Network &network, double learning_rate)
        : network_(network),
          lr_(learning_rate) {}

        double train(const Matrix &inputs, const Matrix &targets, std::size_t epochs) {
            double loss = 0.0;
            for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
                loss = evaluate<true>(inputs, targets);
            }
            return loss;
        }

        template <bool training = false>
        double evaluate(const Matrix &inputs, const Matrix &targets) {
            std::size_t sample_count = static_cast<std::size_t>(inputs.rows()); // 样本数量
            double loss = 0.0;
            for (std::size_t i = 0; i < sample_count; ++i) {
                Vector x = inputs.row(i).transpose();
                Vector y = targets.row(i).transpose();

                Vector prediction;
                if constexpr (training) {
                    prediction = network_.forward(x);
                } else {
                    prediction = network_.predict(x);
                }

                Vector error = prediction - y;                                   // 计算预测值与真实值的误差
                loss += error.squaredNorm() / static_cast<double>(error.size()); // 计算均方误差，然后累加

                if constexpr (training) {
                    Vector grad = (2.0 / static_cast<double>(error.size())) * error;
                    network_.backward(grad, lr_); // 反向传播更新参数
                }
            }
            return loss / static_cast<double>(sample_count);
        }

    private:
        Network &network_;
        double lr_;
    };
} // namespace nn
