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
            Vector out = input;
            for (auto &layer : layers_) {
                out = layer->forward(out);
            }
            return out;
        }

        Vector predict(const Vector &input) {
            return forward(input);
        }

        void backward(const Vector &output, double lr) {
            Vector grad = output;
            for (auto &layer : layers_ | std::ranges::views::reverse) {
                grad = layer->backward(grad, lr);
            }
        }

    private:
        std::vector<std::unique_ptr<Layer>> layers_;
    };
} // namespace nn
