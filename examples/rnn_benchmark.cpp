#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include "rnn.hpp"

int main() {
    global_tape.set_wait_strategy(WaitStrategy::YIELD);
    
    std::thread worker(&Tape::worker_loop, &global_tape);
    
    RNNLayer rnn(16, 64); 
    
    std::vector<std::vector<double>> sequence;
    for (int t = 0; t < 1000; ++t) {
        std::vector<double> xt;
        for (int i = 0; i < 16; ++i) xt.push_back(std::sin(t * 0.1 + i));
        sequence.push_back(xt);
    }

    // ---------------------------------------------
    // RUN 1: STRICT STANDARD AD (BPTT)
    // ---------------------------------------------
    auto p_std = rnn.init_params();
    ADNode loss_std{0.0, global_tape.get_index()};
    
    std::vector<ADNode> h_std;
    for (int i = 0; i < 64; ++i) h_std.push_back({0.0, global_tape.get_index()});

    auto start_std = std::chrono::high_resolution_clock::now();
    for (const auto& xt : sequence) {
        std::vector<ADNode> x_nodes;
        for (double v : xt) x_nodes.push_back({v, global_tape.get_index()});
        
        h_std = rnn.step_standard(x_nodes, h_std, p_std);
        
        ADNode err = std_sub(h_std[0], std::cos(xt[0]));
        loss_std = std_add(loss_std, std_mul(err, err));
    }
    
    std::vector<double> grads_std = global_tape.backward(loss_std.tape_idx);
    auto end_std = std::chrono::high_resolution_clock::now();
    
    size_t mem_std = global_tape.get_memory_bytes();
    size_t nodes_std = global_tape.get_node_count();
    
    // ---------------------------------------------
    // RUN 2: OPTIMIZED AD (BPTT)
    // ---------------------------------------------
    global_tape.clear(); 
    
    auto p_opt = rnn.init_params(); 
    ADNode loss_opt{0.0, global_tape.get_index()};
    
    std::vector<ADNode> h_opt;
    for (int i = 0; i < 64; ++i) h_opt.push_back({0.0, global_tape.get_index()});

    auto start_opt = std::chrono::high_resolution_clock::now();
    for (const auto& xt : sequence) {
        std::vector<ADNode> x_nodes;
        for (double v : xt) x_nodes.push_back({v, global_tape.get_index()});
        
        h_opt = rnn.step_optimized(x_nodes, h_opt, p_opt);
        
        ADNode err = h_opt[0] - std::cos(xt[0]);
        loss_opt = std::move(loss_opt) + (err * err);
    }
    
    global_tape.flush();
    std::vector<double> grads_opt = global_tape.backward(loss_opt.tape_idx);
    auto end_opt = std::chrono::high_resolution_clock::now();
    
    size_t mem_opt = global_tape.get_memory_bytes();
    size_t nodes_opt = global_tape.get_node_count();

    global_tape.stop();
    worker.join();

    // ---------------------------------------------
    // REPORTING
    // ---------------------------------------------
    std::chrono::duration<double, std::milli> time_std = end_std - start_std;
    std::chrono::duration<double, std::milli> time_opt = end_opt - start_opt;

    std::cout << "=== RNN PERFORMANCE METRICS ===\n";
    std::cout << "Standard Runtime:  " << time_std.count() << " ms\n";
    std::cout << "Optimized Runtime: " << time_opt.count() << " ms\n";
    std::cout << "Speedup:           " << time_std.count() / time_opt.count() << "x\n\n";

    std::cout << "=== RNN MEMORY CONSUMPTION ===\n";
    std::cout << "Standard Tape:  " << nodes_std << " nodes | " << mem_std / 1024.0 / 1024.0 << " MB\n";
    std::cout << "Optimized Tape: " << nodes_opt << " nodes | " << mem_opt / 1024.0 / 1024.0 << " MB\n";
    std::cout << "Memory Savings: " << 100.0 * (1.0 - (double)mem_opt / mem_std) << "%\n\n";

    double max_err = 0.0;
    for (size_t i = 0; i < p_opt.W_hh_nodes.size(); ++i) {
        double diff = std::abs(grads_std[p_std.W_hh_nodes[i].tape_idx] - grads_opt[p_opt.W_hh_nodes[i].tape_idx]);
        max_err = std::max(max_err, diff);
    }

    std::cout << "=== CORRECTNESS ===\n";
    std::cout << std::scientific << "Max Gradient Error (W_hh): " << max_err << "\n";

    return 0;
}
