#pragma once
#include <print>
#include <ANN>

using namespace nn;

inline void test1() {
    std::size_t size = 200;
    Matrix inputs(size, 2);
    Matrix targets(size, 1);

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist_x1(0.0, 1.0);
    std::uniform_real_distribution<double> dist_x2(-1.0, 1.0);

    for (std::size_t i = 0; i < size; ++i) {
        double x1 = dist_x1(gen);
        double x2 = dist_x2(gen);
        double y = std::sin(2.0 * std::numbers::pi * x1) + 0.5 * x2 * x2;

        inputs(static_cast<Eigen::Index>(i), 0) = x1;
        inputs(static_cast<Eigen::Index>(i), 1) = x2;
        targets(static_cast<Eigen::Index>(i), 0) = y;
    }

    Network net;
    net.addLayer<DenseLayer>(2, 16, Activation::ReLU);
    net.addLayer<DenseLayer>(16, 16, Activation::Sigmoid);
    net.addLayer<DenseLayer>(16, 1, Activation::None);

    double learning_rate = 0.01;
    Trainer trainer(net, learning_rate);
    trainer.train(inputs, targets, 10000);

    std::println("loss: {}", trainer.evaluate(inputs, targets));

    Vector prediction(2);
    prediction << 0.0, 1.0;

    std::println("predictions: {}", net.predict(prediction));

    std::println("targets:\n{}",
                 std::sin(2.0 * std::numbers::pi * prediction(0)) + 0.5 * prediction(1) * prediction(1));
}