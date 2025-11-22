#pragma once
#include <memory>
#include <utility>
#include <vector>
#include <ranges>
#include "layer.hpp"

namespace nn {
    class Network {
    public:
        Network() = default;

        template <typename LayerType, typename... Args>
        void addLayer(Args &&...args) {
            layers_.emplace_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
        }

        Vector forward(const Vector &input) {
            return run(input, true);
        }

        Vector predict(const Vector &input) {
            return run(input, false);
        }

        void backward(const Vector &output, double lr) {
            Vector grad = output;
            for (auto &layer : layers_ | std::ranges::views::reverse) {
                grad = layer->backward(grad, lr);
            }
        }

    private:
        Vector run(const Vector &input, bool training) {
            Vector out = input;
            for (auto &layer : layers_) {
                out = training ? layer->forward(out) : layer->predict(out);
            }
            return out;
        }

        std::vector<std::unique_ptr<Layer>> layers_;
    };
} // namespace nn
