#pragma once
#include <cstdlib>
#include <print>
#include <huxint/ann>

using namespace nn;

inline void test1() {
    std::size_t size = 200;
    Matrix inputs(size, 2);
    Matrix targets(size, 1);

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist_x1(0.0, 1.0);
    std::uniform_real_distribution<double> dist_x2(-1.0, 1.0);

    auto function = [](const Vector &x) {
        return std::sin(2.0 * std::numbers::pi * x(0)) + 0.5 * x(1) * x(1);
    };

    for (std::size_t i = 0; i < size; ++i) {
        double x1 = dist_x1(gen);
        double x2 = dist_x2(gen);
        Vector x(2);
        x << x1, x2;
        inputs(static_cast<Eigen::Index>(i), 0) = x1;
        inputs(static_cast<Eigen::Index>(i), 1) = x2;
        targets(static_cast<Eigen::Index>(i), 0) = function(x);
    }

    Network net;
    net.addLayer<DenseLayer>(2, 16, Activation::ReLU);
    net.addLayer<DenseLayer>(16, 16, Activation::Sigmoid);
    net.addLayer<DenseLayer>(16, 1, Activation::None);

    double learning_rate = 0.01;
    Trainer trainer(net, learning_rate);
    trainer.train(inputs, targets, 10000);

    Vector prediction(2);
    prediction << dist_x1(gen), dist_x2(gen);

    std::println("loss: {}", trainer.evaluate(inputs, targets));

    auto y1 = function(prediction);
    auto y2 = net.predict(prediction).norm();

    std::println("targets: {}", y1);
    std::println("predictions: {}", y2);
    std::println("{}", std::string(40, '-'));
}

inline void test2() {
    std::size_t size = 2000;

    Matrix inputs(size, 3);
    Matrix targets(size, 1);

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    auto function = [](const Vector &x) {
        return std::sin(std::numbers::pi * x(0)) + 0.5 * x(1) * x(1) + 0.3 * std::cos(3.0 * x(2)) + 0.2 * x(0) * x(2);
    };

    for (std::size_t i = 0; i < size; ++i) {
        double x1 = dist(gen);
        double x2 = dist(gen);
        double x3 = dist(gen);
        Vector x(3);
        x << x1, x2, x3;
        inputs(static_cast<Eigen::Index>(i), 0) = x1;
        inputs(static_cast<Eigen::Index>(i), 1) = x2;
        inputs(static_cast<Eigen::Index>(i), 2) = x3;
        targets(static_cast<Eigen::Index>(i), 0) = function(x);
    }

    // 搭一个层数多一点的网络：3 -> 64 -> 64 -> 32 -> 32 -> 16 -> 1
    Network net;
    net.addLayer<DenseLayer>(3, 64, Activation::ReLU);
    net.addLayer<DenseLayer>(64, 64, Activation::ReLU);
    net.addLayer<DenseLayer>(64, 32, Activation::ReLU);
    net.addLayer<DenseLayer>(32, 32, Activation::ReLU);
    net.addLayer<DenseLayer>(32, 16, Activation::ReLU);
    net.addLayer<DenseLayer>(16, 1, Activation::None); // 最后一层线性输出，适合回归

    double learning_rate = 0.01;
    Trainer trainer(net, learning_rate);
    trainer.train(inputs, targets, 1000);

    Vector prediction(3);
    prediction << dist(gen), dist(gen), dist(gen);

    auto y1 = function(prediction);
    auto y2 = net.predict(prediction).norm();

    std::println("loss: {}", trainer.evaluate(inputs, targets));
    std::println("targets: {}", function(prediction));
    std::println("predictions: {}", net.predict(prediction));
    std::println("{}", std::string(40, '-'));
}