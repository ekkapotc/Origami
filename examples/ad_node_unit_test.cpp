#include <iostream>
#include <iomanip>
#include <cmath>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "ad_node.hpp"
#include "tape.hpp"

// Verifies analytic gradients produced by the tape (including the rvalue
// "fusion" operator paths, where a derivative is folded into an existing
// node's edges instead of allocating a new node) against gradients obtained
// via central finite differences on an independent reference function.
struct GradCheck {
    std::string name;
    std::function<ADNode(std::vector<ADNode>&)> build;
    std::function<double(const std::vector<double>&)> ref;
    std::vector<double> x0;
};

bool run_check(const GradCheck& tc, double h = 1e-6, double tol = 1e-6) {
    global_tape.clear();

    std::vector<ADNode> nodes;
    for (double v : tc.x0) nodes.push_back({v, global_tape.get_index()});

    ADNode out = tc.build(nodes);
    global_tape.flush();
    std::vector<double> grads = global_tape.backward(out.tape_idx);

    double expected_val = tc.ref(tc.x0);
    double val_err = std::abs(out.value - expected_val);

    double max_grad_err = 0.0;
    for (size_t i = 0; i < tc.x0.size(); ++i) {
        std::vector<double> xp = tc.x0, xm = tc.x0;
        xp[i] += h;
        xm[i] -= h;
        double numeric = (tc.ref(xp) - tc.ref(xm)) / (2.0 * h);
        double analytic = grads[nodes[i].tape_idx];
        max_grad_err = std::max(max_grad_err, std::abs(numeric - analytic));
    }

    bool passed = val_err < tol && max_grad_err < tol;

    std::cout << std::left << std::setw(38) << tc.name
               << (passed ? "PASS" : "FAIL")
               << "  value_err=" << std::scientific << std::setprecision(2) << val_err
               << "  max_grad_err=" << max_grad_err << "\n";

    return passed;
}

int main() {
    std::thread worker(&Tape::worker_loop, &global_tape);

    std::vector<GradCheck> cases;

    // Scalar operators chained through rvalue-fusing overloads:
    // r = (10 - ((x + 2) * 3)) / 2
    cases.push_back({
        "scalar_ops_rvalue_chain",
        [](std::vector<ADNode>& n) {
            ADNode a = n[0];
            ADNode r1 = a + 2.0;
            ADNode r2 = std::move(r1) * 3.0;
            ADNode r3 = 10.0 - std::move(r2);
            return std::move(r3) / 2.0;
        },
        [](const std::vector<double>& x) {
            return (10.0 - ((x[0] + 2.0) * 3.0)) / 2.0;
        },
        {0.85}
    });

    // Node-node operators with rvalue fusion: r = x - (x*y + y)
    cases.push_back({
        "node_node_ops_rvalue_fusion",
        [](std::vector<ADNode>& n) {
            ADNode a = n[0], b = n[1];
            ADNode p = a * b;
            ADNode q = std::move(p) + b;
            return a - std::move(q);
        },
        [](const std::vector<double>& x) {
            return x[0] - (x[0] * x[1] + x[1]);
        },
        {1.3, -0.7}
    });

    // Division in both directions, lvalue and rvalue forms: r = x / ((x/y)/y)
    // which algebraically collapses to y^2 (zero gradient w.r.t. x).
    cases.push_back({
        "division_both_directions",
        [](std::vector<ADNode>& n) {
            ADNode a = n[0], b = n[1];
            ADNode p = a / b;
            ADNode q = std::move(p) / b;
            return a / std::move(q);
        },
        [](const std::vector<double>& x) {
            return x[0] / ((x[0] / x[1]) / x[1]);
        },
        {2.0, 0.5}
    });

    // Transcendental chain through rvalue-fusing overloads:
    // r = log(exp(cos(sin(x))))
    cases.push_back({
        "transcendental_rvalue_chain",
        [](std::vector<ADNode>& n) {
            ADNode a = n[0];
            ADNode s = sin(a);
            ADNode c = cos(std::move(s));
            ADNode e = exp(std::move(c));
            return log(std::move(e));
        },
        [](const std::vector<double>& x) {
            return std::log(std::exp(std::cos(std::sin(x[0]))));
        },
        {0.4}
    });

    // tanh with mixed lvalue/rvalue scalar ops: r = tanh(2x - 1)^2
    cases.push_back({
        "tanh_mixed_lvalue_rvalue",
        [](std::vector<ADNode>& n) {
            ADNode a = n[0];
            ADNode z = (a * 2.0) - 1.0;
            ADNode t = tanh(std::move(z));
            return t * t;
        },
        [](const std::vector<double>& x) {
            double t = std::tanh(2.0 * x[0] - 1.0);
            return t * t;
        },
        {0.55}
    });

    std::cout << "=== AD_NODE GRADIENT CHECKS ===\n";
    bool all_passed = true;
    for (const auto& tc : cases) {
        if (!run_check(tc)) all_passed = false;
    }

    global_tape.stop();
    worker.join();

    std::cout << "\n" << (all_passed ? "All gradient checks passed." : "Some gradient checks FAILED.") << "\n";

    return all_passed ? 0 : 1;
}
