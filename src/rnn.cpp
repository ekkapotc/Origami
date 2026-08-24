#include "rnn.hpp"

RNNLayer::RNNLayer(int in_d, int h_d) : in_dim(in_d), h_dim(h_d) {
    std::mt19937 gen(42);
    std::normal_distribution<double> d(0.0, std::sqrt(2.0 / (in_dim + h_dim)));
    for (int i = 0; i < in_dim * h_dim; ++i) W_xh.push_back(d(gen));
    for (int i = 0; i < h_dim * h_dim; ++i) W_hh.push_back(d(gen));
    for (int i = 0; i < h_dim; ++i) b_h.push_back(0.0);
}

RNNLayer::LayerParams RNNLayer::init_params() const {
    LayerParams p;
    for (double w : W_xh) p.W_xh_nodes.push_back({w, global_tape.get_index()});
    for (double w : W_hh) p.W_hh_nodes.push_back({w, global_tape.get_index()});
    for (double b : b_h) p.b_nodes.push_back({b, global_tape.get_index()});
    return p;
}

std::vector<ADNode> RNNLayer::step_standard(const std::vector<ADNode>& x, 
                                            const std::vector<ADNode>& h_prev, 
                                            const LayerParams& params) {
    std::vector<ADNode> h_next;
    for (int j = 0; j < h_dim; ++j) {
        ADNode acc = std_add(std_mul(x[0], params.W_xh_nodes[j]), std_mul(h_prev[0], params.W_hh_nodes[j]));
        
        for (int i = 1; i < in_dim; ++i) {
            acc = std_add(acc, std_mul(x[i], params.W_xh_nodes[i * h_dim + j]));
        }
        for (int i = 1; i < h_dim; ++i) {
            acc = std_add(acc, std_mul(h_prev[i], params.W_hh_nodes[i * h_dim + j]));
        }
        acc = std_add(acc, params.b_nodes[j]);
        h_next.push_back(std_tanh(acc));
    }
    return h_next;
}

std::vector<ADNode> RNNLayer::step_optimized(const std::vector<ADNode>& x, 
                                             const std::vector<ADNode>& h_prev, 
                                             const LayerParams& params) {
    std::vector<ADNode> h_next;
    for (int j = 0; j < h_dim; ++j) {
        ADNode acc = (x[0] * params.W_xh_nodes[j]) + (h_prev[0] * params.W_hh_nodes[j]);
        
        for (int i = 1; i < in_dim; ++i) {
            acc = std::move(acc) + (x[i] * params.W_xh_nodes[i * h_dim + j]);
        }
        for (int i = 1; i < h_dim; ++i) {
            acc = std::move(acc) + (h_prev[i] * params.W_hh_nodes[i * h_dim + j]);
        }
        acc = std::move(acc) + params.b_nodes[j];
        h_next.push_back(tanh(std::move(acc)));
    }
    return h_next;
}
