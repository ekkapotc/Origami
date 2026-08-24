#include "dense.hpp"
#include <random>

DenseLayer::DenseLayer(int in_d, int out_d) : in_dim(in_d), out_dim(out_d) {
        std::mt19937 gen(42);
        std::normal_distribution<double> d(0.0, std::sqrt(2.0 / in_dim));
        for (int i = 0; i < in_dim * out_dim; ++i) weights.push_back(d(gen));
        for (int i = 0; i < out_dim; ++i) biases.push_back(0.0);
}

LayerParams DenseLayer::init_params() const {
        LayerParams p;
        for (double w : weights) p.w_nodes.push_back({w, global_tape.get_index()});
        for (double b : biases) p.b_nodes.push_back({b, global_tape.get_index()});
        return p;
}

std::vector<ADNode> DenseLayer::forward_standard(const std::vector<ADNode>& in, const LayerParams& params, bool act) {
        std::vector<ADNode> out;
        for (int j = 0; j < out_dim; ++j) {
            ADNode acc = in[0] * params.w_nodes[j];
            for (int i = 1; i < in_dim; ++i) {
                acc = acc + (in[i] * params.w_nodes[i * out_dim + j]);
            }
            acc = acc + params.b_nodes[j];
            out.push_back(act ? tanh(acc) : acc);
        }
        return out;
}

std::vector<ADNode> DenseLayer::forward(const std::vector<ADNode>& in, const LayerParams& params, bool act) {
        std::vector<ADNode> out;
        for (int j = 0; j < out_dim; ++j) {
            ADNode acc = in[0] * params.w_nodes[j];
            for (int i = 1; i < in_dim; ++i) {
                acc = std::move(acc) + (in[i] * params.w_nodes[i * out_dim + j]);
            }
            acc = std::move(acc) + params.b_nodes[j];
            out.push_back(act ? tanh(std::move(acc)) : acc);
        }
        return out;
}


