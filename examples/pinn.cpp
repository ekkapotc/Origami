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

// Estimated node size in bytes for tape memory calculation
constexpr size_t NODE_SIZE_BYTES = 48;

inline ADNode make_node(double val) {
    return ADNode{val, global_tape.get_index()};
}

// --- DUAL NODE DEFINITION ---
struct PINNode { ADNode val; ADNode dx; ADNode dxx; };

// Standard Dual-Activation (No Move Semantics)
PINNode tanh_pinn_standard(const PINNode& z) {
    ADNode th = tanh(z.val);
    ADNode dt = 1.0 - (th * th);
    ADNode d2t = (th * dt) * make_node(-2.0);
    
    return {
        th,
        dt * z.dx,
        (d2t * (z.dx * z.dx)) + (dt * z.dxx)
    };
}

// Optimized Dual-Activation (With Move Semantics)
PINNode tanh_pinn(PINNode&& z) {
    ADNode th = tanh(z.val);
    ADNode dt = 1.0 - (th * th);
    ADNode d2t = (th * dt) * make_node(-2.0);
    
    return {
        std::move(th),
        dt * z.dx,
        (d2t * (z.dx * z.dx)) + (dt * z.dxx)
    };
}

// --- PINN MODEL ---
struct PINN {
    int hidden_dim = 16;
    std::vector<double> w1, b1, w2;
    double b2 = 0.0;

    PINN() {
        std::mt19937 gen(42);
        std::normal_distribution<double> d(0.0, 0.5);
        for (int i = 0; i < hidden_dim; ++i) {
            w1.push_back(d(gen));
            b1.push_back(0.0);
            w2.push_back(d(gen));
        }
    }

    PINNode forward_standard(double x_val, const std::vector<ADNode>& p_w1, const std::vector<ADNode>& p_b1, 
                             const std::vector<ADNode>& p_w2, const ADNode& p_b2) {
        PINNode x_in{make_node(x_val), make_node(1.0), make_node(0.0)};
        
        std::vector<PINNode> h;
        for (int j = 0; j < hidden_dim; ++j) {
            PINNode z{
                (p_w1[j] * x_in.val) + p_b1[j],
                p_w1[j] * x_in.dx,
                p_w1[j] * x_in.dxx
            };
            h.push_back(tanh_pinn_standard(z));
        }

        PINNode out{p_b2, make_node(0.0), make_node(0.0)};
        for (int j = 0; j < hidden_dim; ++j) {
            out.val = out.val + (p_w2[j] * h[j].val);
            out.dx  = out.dx  + (p_w2[j] * h[j].dx);
            out.dxx = out.dxx + (p_w2[j] * h[j].dxx);
        }
        return out;
    }

    PINNode forward(double x_val, const std::vector<ADNode>& p_w1, const std::vector<ADNode>& p_b1, 
                    const std::vector<ADNode>& p_w2, const ADNode& p_b2) {
        PINNode x_in{make_node(x_val), make_node(1.0), make_node(0.0)};
        
        std::vector<PINNode> h;
        for (int j = 0; j < hidden_dim; ++j) {
            PINNode z{
                (p_w1[j] * x_in.val) + p_b1[j],
                p_w1[j] * x_in.dx,
                p_w1[j] * x_in.dxx
            };
            h.push_back(tanh_pinn(std::move(z)));
        }

        PINNode out{p_b2, make_node(0.0), make_node(0.0)};
        for (int j = 0; j < hidden_dim; ++j) {
            out.val = std::move(out.val) + (p_w2[j] * h[j].val);
            out.dx  = std::move(out.dx)  + (p_w2[j] * h[j].dx);
            out.dxx = std::move(out.dxx) + (p_w2[j] * h[j].dxx);
        }
        return out;
    }
};

int main() {
    std::thread worker(&Tape::worker_loop, &global_tape);
    
    PINN model_std;
    PINN model_opt = model_std; // Identical initial weights

    const double PI = 3.141592653589793;
    const double lr = 0.001;
    const int num_epochs = 500;

    size_t std_nodes_per_epoch = 0;
    size_t opt_nodes_per_epoch = 0;

    std::vector<double> grads_std_first, grads_opt_first;
    std::vector<ADNode> p1_w1_first, p1_b1_first, p1_w2_first;
    ADNode p1_b2_first{0.0, 0};

    std::vector<ADNode> p2_w1_first, p2_b1_first, p2_w2_first;
    ADNode p2_b2_first{0.0, 0};

    // ---------------------------------------------
    // RUN 1: STANDARD AD
    // ---------------------------------------------
    auto start_std = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        global_tape.clear();

        std::vector<ADNode> p_w1, p_b1, p_w2;
        for (double w : model_std.w1) p_w1.push_back(make_node(w));
        for (double b : model_std.b1) p_b1.push_back(make_node(b));
        for (double w : model_std.w2) p_w2.push_back(make_node(w));
        ADNode p_b2 = make_node(model_std.b2);

        PINNode u_0  = model_std.forward_standard(0.0, p_w1, p_b1, p_w2, p_b2);
        PINNode u_pi = model_std.forward_standard(PI / 2.0, p_w1, p_b1, p_w2, p_b2);
        ADNode loss_bc = (u_0.val * u_0.val) + ((u_pi.val - make_node(1.0)) * (u_pi.val - make_node(1.0)));

        ADNode loss_pde = make_node(0.0);
        int num_pts = 10;
        for (int i = 1; i < num_pts; ++i) {
            double x = (PI / 2.0) * (i / (double)num_pts);
            PINNode u_x = model_std.forward_standard(x, p_w1, p_b1, p_w2, p_b2);
            ADNode res = u_x.dxx + u_x.val;
            loss_pde = loss_pde + (res * res);
        }

        ADNode total_loss = (loss_pde * 0.1) + (loss_bc * 10.0);
        
        if (epoch == 0) std_nodes_per_epoch = global_tape.get_index();

        global_tape.flush();
        std::vector<double> grads = global_tape.backward(total_loss.tape_idx);

        if (epoch == 0) {
            grads_std_first = grads;
            p1_w1_first = p_w1; p1_b1_first = p_b1;
            p1_w2_first = p_w2; p1_b2_first = p_b2;
        }

        for (int j = 0; j < model_std.hidden_dim; ++j) {
            model_std.w1[j] -= lr * grads[p_w1[j].tape_idx];
            model_std.b1[j] -= lr * grads[p_b1[j].tape_idx];
            model_std.w2[j] -= lr * grads[p_w2[j].tape_idx];
        }
        model_std.b2 -= lr * grads[p_b2.tape_idx];
    }

    auto end_std = std::chrono::high_resolution_clock::now();
    double time_std_ms = std::chrono::duration<double, std::milli>(end_std - start_std).count();

    // ---------------------------------------------
    // RUN 2: PREACCUMULATED / OPTIMIZED AD
    // ---------------------------------------------
    auto start_opt = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        global_tape.clear();

        std::vector<ADNode> p_w1, p_b1, p_w2;
        for (double w : model_opt.w1) p_w1.push_back(make_node(w));
        for (double b : model_opt.b1) p_b1.push_back(make_node(b));
        for (double w : model_opt.w2) p_w2.push_back(make_node(w));
        ADNode p_b2 = make_node(model_opt.b2);

        PINNode u_0  = model_opt.forward(0.0, p_w1, p_b1, p_w2, p_b2);
        PINNode u_pi = model_opt.forward(PI / 2.0, p_w1, p_b1, p_w2, p_b2);
        ADNode loss_bc = (u_0.val * u_0.val) + ((u_pi.val - make_node(1.0)) * (u_pi.val - make_node(1.0)));

        ADNode loss_pde = make_node(0.0);
        int num_pts = 10;
        for (int i = 1; i < num_pts; ++i) {
            double x = (PI / 2.0) * (i / (double)num_pts);
            PINNode u_x = model_opt.forward(x, p_w1, p_b1, p_w2, p_b2);
            ADNode res = u_x.dxx + u_x.val;
            loss_pde = std::move(loss_pde) + (res * res);
        }

        ADNode total_loss = (loss_pde * 0.1) + (loss_bc * 10.0);
        
        if (epoch == 0) opt_nodes_per_epoch = global_tape.get_index();

        global_tape.flush();
        std::vector<double> grads = global_tape.backward(total_loss.tape_idx);

        if (epoch == 0) {
            grads_opt_first = grads;
            p2_w1_first = p_w1; p2_b1_first = p_b1;
            p2_w2_first = p_w2; p2_b2_first = p_b2;
        }

        for (int j = 0; j < model_opt.hidden_dim; ++j) {
            model_opt.w1[j] -= lr * grads[p_w1[j].tape_idx];
            model_opt.b1[j] -= lr * grads[p_b1[j].tape_idx];
            model_opt.w2[j] -= lr * grads[p_w2[j].tape_idx];
        }
        model_opt.b2 -= lr * grads[p_b2.tape_idx];
    }

    auto end_opt = std::chrono::high_resolution_clock::now();
    double time_opt_ms = std::chrono::duration<double, std::milli>(end_opt - start_opt).count();

    global_tape.stop();
    worker.join();

    // ---------------------------------------------
    // CORRECTNESS CHECK
    // ---------------------------------------------
    double max_grad_err = 0.0;
    for (int j = 0; j < model_std.hidden_dim; ++j) {
        max_grad_err = std::max(max_grad_err, std::abs(grads_std_first[p1_w1_first[j].tape_idx] - grads_opt_first[p2_w1_first[j].tape_idx]));
        max_grad_err = std::max(max_grad_err, std::abs(grads_std_first[p1_b1_first[j].tape_idx] - grads_opt_first[p2_b1_first[j].tape_idx]));
        max_grad_err = std::max(max_grad_err, std::abs(grads_std_first[p1_w2_first[j].tape_idx] - grads_opt_first[p2_w2_first[j].tape_idx]));
    }
    max_grad_err = std::max(max_grad_err, std::abs(grads_std_first[p1_b2_first.tape_idx] - grads_opt_first[p2_b2_first.tape_idx]));

    // ---------------------------------------------
    // METRIC CALCULATIONS
    // ---------------------------------------------
    double std_mem_mb = (std_nodes_per_epoch * NODE_SIZE_BYTES) / (1024.0 * 1024.0);
    double opt_mem_mb = (opt_nodes_per_epoch * NODE_SIZE_BYTES) / (1024.0 * 1024.0);

    double speedup = time_std_ms / time_opt_ms;
    double mem_savings_pct = (1.0 - (double)opt_nodes_per_epoch / (double)std_nodes_per_epoch) * 100.0;

    // ---------------------------------------------
    // PRINT REPORT
    // ---------------------------------------------
    std::cout << "\n=== PINN PERFORMANCE METRICS ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Standard Runtime:  " << time_std_ms << " ms\n";
    std::cout << "Optimized Runtime: " << time_opt_ms << " ms\n";
    std::cout << std::setprecision(4);
    std::cout << "Speedup:           " << speedup << "x\n\n";

    std::cout << "=== PINN MEMORY CONSUMPTION ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Standard Tape:  " << std_nodes_per_epoch << " nodes | " << std_mem_mb << " MB\n";
    std::cout << "Optimized Tape: " << opt_nodes_per_epoch << " nodes | " << opt_mem_mb << " MB\n";
    std::cout << std::setprecision(4);
    std::cout << "Memory Savings: " << mem_savings_pct << "%\n\n";

    std::cout << "=== CORRECTNESS ===\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Max Gradient Error (PINN Weights): " << max_grad_err << "\n";

    return 0;
}
