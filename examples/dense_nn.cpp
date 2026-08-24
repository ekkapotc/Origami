#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <algorithm>

#include "ad_node.hpp"
#include "tape.hpp"
#include "dense.hpp"

// Estimated node size in bytes for tape memory calculation
constexpr size_t NODE_SIZE_BYTES = 48;

int main() {
    std::thread worker(&Tape::worker_loop, &global_tape);
    
    // 3-Layer Network Setup
    DenseLayer l1(10, 32);
    DenseLayer l2(32, 32);
    DenseLayer l3(32, 1);
    
    std::vector<double> t_dataset;
    for (int i = 0; i < 5000; ++i) t_dataset.push_back(i * 0.01);

    // ---------------------------------------------
    // RUN 1: STANDARD AD
    // ---------------------------------------------
    auto start_std = std::chrono::high_resolution_clock::now();

    auto p1_std = l1.init_params();
    auto p2_std = l2.init_params();
    auto p3_std = l3.init_params();

    ADNode loss_std{0.0, global_tape.get_index()};
    
    for (double t : t_dataset) {
        std::vector<ADNode> in;
        for (int i = 0; i < 10; ++i) in.push_back({std::sin(t + i), global_tape.get_index()});
        
        auto h1 = l1.forward_standard(in, p1_std, true);
        auto h2 = l2.forward_standard(h1, p2_std, true);
        auto out = l3.forward_standard(h2, p3_std, false);
        
        ADNode err = out[0] - std::cos(2.0 * t);
        loss_std = loss_std + (err * err);
    }
    
    global_tape.flush();
    std::vector<double> grads_std = global_tape.backward(loss_std.tape_idx);
    
    auto end_std = std::chrono::high_resolution_clock::now();
    double time_std_ms = std::chrono::duration<double, std::milli>(end_std - start_std).count();

    size_t std_nodes = global_tape.get_index();
    double std_mem_mb = (std_nodes * NODE_SIZE_BYTES) / (1024.0 * 1024.0);

    global_tape.clear();

    // ---------------------------------------------
    // RUN 2: PREACCUMULATED / OPTIMIZED AD
    // ---------------------------------------------
    auto start_opt = std::chrono::high_resolution_clock::now();

    auto p1_opt = l1.init_params();
    auto p2_opt = l2.init_params();
    auto p3_opt = l3.init_params();
    
    ADNode loss_opt{0.0, global_tape.get_index()};
    
    for (double t : t_dataset) {
        std::vector<ADNode> in;
        for (int i = 0; i < 10; ++i) in.push_back({std::sin(t + i), global_tape.get_index()});
        
        auto h1 = l1.forward(in, p1_opt, true);
        auto h2 = l2.forward(h1, p2_opt, true);
        auto out = l3.forward(h2, p3_opt, false);
        
        ADNode err = out[0] - std::cos(2.0 * t);
        loss_opt = std::move(loss_opt) + (err * err);
    }
    
    global_tape.flush();
    std::vector<double> grads_opt = global_tape.backward(loss_opt.tape_idx);
    
    auto end_opt = std::chrono::high_resolution_clock::now();
    double time_opt_ms = std::chrono::duration<double, std::milli>(end_opt - start_opt).count();

    size_t opt_nodes = global_tape.get_index();
    double opt_mem_mb = (opt_nodes * NODE_SIZE_BYTES) / (1024.0 * 1024.0);

    global_tape.stop();
    worker.join();

    // ---------------------------------------------
    // CORRECTNESS CHECK (MAX GRADIENT ERROR)
    // ---------------------------------------------
    double max_grad_err = 0.0;
    
    auto check_params = [&](const LayerParams& std_p, const LayerParams& opt_p) {
        for (size_t i = 0; i < std_p.w_nodes.size(); ++i) {
            double err = std::abs(grads_std[std_p.w_nodes[i].tape_idx] - grads_opt[opt_p.w_nodes[i].tape_idx]);
            max_grad_err = std::max(max_grad_err, err);
        }
        for (size_t i = 0; i < std_p.b_nodes.size(); ++i) {
            double err = std::abs(grads_std[std_p.b_nodes[i].tape_idx] - grads_opt[opt_p.b_nodes[i].tape_idx]);
            max_grad_err = std::max(max_grad_err, err);
        }
    };

    check_params(p1_std, p1_opt);
    check_params(p2_std, p2_opt);
    check_params(p3_std, p3_opt);

    // ---------------------------------------------
    // METRIC CALCULATIONS
    // ---------------------------------------------
    double speedup = time_std_ms / time_opt_ms;
    double mem_savings_pct = (1.0 - (double)opt_nodes / (double)std_nodes) * 100.0;

    // ---------------------------------------------
    // PRINT REPORT
    // ---------------------------------------------
    std::cout << "\n=== DENSE PERFORMANCE METRICS ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Standard Runtime:  " << time_std_ms << " ms\n";
    std::cout << "Optimized Runtime: " << time_opt_ms << " ms\n";
    std::cout << std::setprecision(4);
    std::cout << "Speedup:           " << speedup << "x\n\n";

    std::cout << "=== DENSE MEMORY CONSUMPTION ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Standard Tape:  " << std_nodes << " nodes | " << std_mem_mb << " MB\n";
    std::cout << "Optimized Tape: " << opt_nodes << " nodes | " << opt_mem_mb << " MB\n";
    std::cout << std::setprecision(4);
    std::cout << "Memory Savings: " << mem_savings_pct << "%\n\n";

    std::cout << "=== CORRECTNESS ===\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Max Gradient Error (Dense Weights): " << max_grad_err << "\n";

    return 0;
}
