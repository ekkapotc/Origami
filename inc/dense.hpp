#pragma once

#include <vector>

#include "ad_node.hpp"

struct LayerParams { 
	std::vector<ADNode> w_nodes;
	std::vector<ADNode> b_nodes; 
};

struct DenseLayer{
   int in_dim;
   int out_dim;
   
   std::vector<double> weights;
   std::vector<double> biases;
      
   DenseLayer(int in_d, int out_d);

   struct LayerParams init_params() const;

   std::vector<ADNode> forward_standard(const std::vector<ADNode> & , const LayerParams& params, bool act = true);

   std::vector<ADNode> forward(const std::vector<ADNode>& in, const LayerParams& params, bool act = true); 

};
