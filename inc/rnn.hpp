#pragma once
#include <vector>
#include <random>
#include <cmath>
#include "ad_node.hpp"

struct RNNLayer {
    int in_dim, h_dim;
    std::vector<double> W_xh, W_hh, b_h;
    struct LayerParams { 
        std::vector<ADNode> W_xh_nodes, W_hh_nodes, b_nodes; 
    };

    RNNLayer(int in_d, int h_d);

    LayerParams init_params() const;

    std::vector<ADNode> step_standard(const std::vector<ADNode>& x, 
                                     const std::vector<ADNode>& h_prev, 
                                     const LayerParams& params);

    std::vector<ADNode> step_optimized(const std::vector<ADNode>& x, 
                                      const std::vector<ADNode>& h_prev, 
                                      const LayerParams& params);
};
